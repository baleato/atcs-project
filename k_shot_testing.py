import os
from itertools import chain

import copy
from transformers import BertTokenizer, AdamW

from util import get_test_args, get_pytorch_device, load_model
from tasks import *
from models import MultiTaskLearner, PrototypeLearner, ProtoMAMLLearner

import torch.optim as optim
import torch.nn as nn


def k_shot_testing(model, episodes, test_task, device, num_updates=5, num_test_batches=None, lr=1e-3, bert_lr=5e-5, zero_init=False, save_pred=None, path="predictions", init_linear_with_centroids=False):

    if not os.path.exists(path):
        os.makedirs(path)

    # save initial state of the model
    if isinstance(model, ProtoMAMLLearner):
        initial_state = copy.deepcopy(model.proto_net.state_dict())
    else:
        initial_state = copy.deepcopy(model.state_dict())

    # get iterator over test task
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    test_iter = test_task.get_iter('test', tokenizer, shuffle=False)

    # set number of test batches to evaluate on
    test_size = len(test_iter)
    if num_test_batches is None or num_test_batches > test_size:
        num_test_batches = test_size

    # Define optimizers and loss function
    if isinstance(model, MultiTaskLearner):
        task_module_name = 'task_{}'.format(test_task.get_name())
        out_MTL_layer = model._modules[task_module_name]
        optimizer = optim.SGD(params=chain(model.encoder.mlp.parameters(),
                                           out_MTL_layer.parameters()), lr=lr)
        optimizer_bert = optim.SGD(model.encoder.bert.parameters(), lr=bert_lr)
    elif isinstance(model, ProtoMAMLLearner):
        optimizer = optim.SGD(params=chain(model.proto_net.encoder.mlp.parameters(),
                                           model.output_layer.parameters()), lr=lr)
        optimizer_bert = optim.SGD(model.proto_net.encoder.bert.parameters(), lr=bert_lr)
    else:
        optimizer = optim.SGD(params=model.encoder.mlp.parameters(), lr=lr)
        optimizer_bert = optim.SGD(model.encoder.bert.parameters(), lr=bert_lr)

    cross_entropy = nn.CrossEntropyLoss()

    model.to(device)

    episode_accs = []
    for episode in episodes:
        # reset model to initial state
        if isinstance(model, ProtoMAMLLearner):
            model.proto_net.load_state_dict(copy.deepcopy(initial_state))
        else:
            model.load_state_dict(copy.deepcopy(initial_state))
        model.train()
        # setup output layer for ProtoMAML and MTL
        if isinstance(model, ProtoMAMLLearner):
            proto_embeddings = model.proto_net(episode[0].to(device), attention_mask=episode[2].to(device))
            prototypes = model.proto_net.calculate_centroids((proto_embeddings, episode[1]), test_task.num_classes)
            W, b = model.calculate_output_params(prototypes.detach())
            model.initialize_classifier(W, b)
        elif init_linear_with_centroids and isinstance(model, MultiTaskLearner):
            print('Initialise with centroids')
            proto_embeddings = model.encoder(episode[0].to(device), attention_mask=episode[2].to(device))
            _self = None
            prototypes = PrototypeLearner.calculate_centroids(_self, (proto_embeddings, episode[1]), test_task.num_classes)
            W, b = ProtoMAMLLearner.calculate_output_params(_self, prototypes.detach())
            out_MTL_layer.network[0].weight.data = W
            out_MTL_layer.network[0].bias.data = b
        elif zero_init and isinstance(model, MultiTaskLearner):
            out_MTL_layer.weight.data = torch.zeros_like(out_MTL_layer.weight.data)
            out_MTL_layer.bias.data = torch.zeros_like(out_MTL_layer.bias.data)

        # fine-tune model with some updates on the provided episode
        for update in range(num_updates):
            optimizer_bert.zero_grad()
            optimizer.zero_grad()

            # get predictions depending on model type
            if isinstance(model, MultiTaskLearner):
                predictions = model(episode[0].to(device), test_task.get_name(), attention_mask=episode[2].to(device))
            else:
                predictions = model(episode[0].to(device), attention_mask=episode[2].to(device))

            # compute loss depending on model type
            if isinstance(model, PrototypeLearner):
                centroids = model.calculate_centroids((predictions, episode[1]), test_task.num_classes)
                distances = model.compute_distance(predictions, centroids)
                loss = cross_entropy(-distances, episode[1].long().squeeze().to(device))
            else:
                loss = cross_entropy(predictions, episode[1].long().squeeze().to(device))

            loss.backward()
            optimizer.step()
            optimizer_bert.step()


        # evaluate accuracy on whole test set
        with torch.no_grad():
            model.eval()
            accuracies = []
            preds = {}
            preds['raw'] = []
            preds['class'] = []
            batches_tested = 0
            for batch in test_iter:
                if isinstance(model, MultiTaskLearner):
                    predictions = model(batch[0].to(device), test_task.get_name(), attention_mask=batch[2].to(device))
                elif isinstance(model, PrototypeLearner):
                    predictions = model(batch[0].to(device), attention_mask=batch[2].to(device))
                    predictions = -model.compute_distance(predictions, centroids)
                else:
                    predictions = model(batch[0].to(device), attention_mask=batch[2].to(device))
                acc = test_task.calculate_accuracy(predictions, batch[1].to(device))
                accuracies.append(acc)
                if save_pred is not None:
                    preds['raw'].append(predictions)
                    preds['class'].append(predictions.argmax(dim=1, keepdim=False))
                batches_tested += 1
                if batches_tested == num_test_batches:
                    break
        episode_accs.append(np.asarray(accuracies).mean())
        if save_pred is not None:
            preds['raw'] = torch.cat(preds['raw'], dim=0).cpu()
            preds['class'] = torch.cat(preds['class'], dim=0).cpu()
            # save predictions
            preds_path = 'predictions_{}_{}.pkl'.format(save_pred, task.get_name())
            torch.save(preds, open(os.path.join(path, preds_path), "wb"))

    return np.asarray(episode_accs).mean(), np.asarray(episode_accs).std()

def sample_episodes(k, task, tokenizer, num_episodes=10):
    # setup sampler
    episode_generator = iter(task.get_iter('train', tokenizer, batch_size=k * task.num_classes, shuffle=True))
    # generate episodes
    episodes = []
    for i in range(num_episodes):
        episodes.append(next(episode_generator))
    return episodes

if __name__ == '__main__':
    args = get_test_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)

    os.makedirs(args.save_path, exist_ok=True)

    if args.task == 'SentimentAnalysis':
        task = SentimentAnalysis(cls_dim=args.mlp_dims[-1])
    elif args.task == 'IronySubtaskA':
        task = IronySubtaskA(cls_dim=args.mlp_dims[-1])
    elif args.task == 'IronySubtaskB':
        task = IronySubtaskB(cls_dim=args.mlp_dims[-1])
    elif args.task == 'Abuse':
        task = Abuse(cls_dim=args.mlp_dims[-1])
    elif args.task == "SarcasmDetection":
        task = SarcasmDetection(cls_dim=args.mlp_dims[-1])
    elif args.task == "OffenseEval":
        task = OffensevalTask(cls_dim=args.mlp_dims[-1])
    elif args.task == 'Politeness':
        task = Politeness(cls_dim=args.mlp_dims[-1])
    else:
        task = None
        RuntimeError('Unknown evaluation task!')

    print("Loading model from snapshot")
    if args.model == 'MTL':
        model = MultiTaskLearner(args)
    elif args.model == 'ProtoNet':
        model = PrototypeLearner(args)
    elif args.model == 'ProtoMAML':
        model = ProtoMAMLLearner(args)
    else:
        model = None
        RuntimeError('Unknown model type!')
    model.load_model(args.model_path, device)
    if isinstance(model, MultiTaskLearner):
        model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if not args.episodes == '':
        episodes = torch.load(open(args.episodes, "rb"), map_location=device)
    else:
        episodes = sample_episodes(args.k, task, tokenizer, args.generate_episodes)
        random_id = int(np.random.randint(0, 10000, 1))
        torch.save(episodes, open(args.save_path+"/episodes_{}.pkl".format(random_id), "wb"))

    mean, stddev = k_shot_testing(model, episodes, task, device, args.num_updates, args.num_test_batches,
                                  lr=args.lr, bert_lr=args.bert_lr, save_pred=args.model, init_linear_with_centroids=args.init_linear_with_centroids)
    print("Mean accuracy: {}, standard deviation: {}\t{:.2f} +/- {:.1f}".format(mean, stddev, mean * 100, stddev * 100))

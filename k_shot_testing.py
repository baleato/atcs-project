import os
from itertools import chain


from transformers import BertTokenizer, AdamW

from util import get_test_args, get_pytorch_device, load_model
from tasks import *
from models import MultiTaskLearner, PrototypeLearner, ProtoMAMLLearner

import torch.optim as optim
import torch.nn as nn


def k_shot_testing(model, episodes, test_task, device, num_updates=5, num_test_batches=None, lr=1e-3, zero_init=False):
    # get iterator over test task
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    test_iter = test_task.get_iter('test', tokenizer, shuffle=False)

    # set number of test batches to evaluate on
    test_size = len(test_iter)
    if num_test_batches is None or num_test_batches > test_size:
        num_test_batches = test_size

    # Define optimizers and loss function
    if isinstance(model, ProtoMAMLLearner):
        optimizer_bert = optim.SGD(model.proto_net.encoder.bert.parameters(), lr=args.bert_lr)
    else:
        optimizer_bert = optim.SGD(model.encoder.bert.parameters(), lr=args.bert_lr)
    if isinstance(model, MultiTaskLearner):
        task_module_name = 'task_{}'.format(test_task.get_name())
        out_MTL_layer = model._modules[task_module_name]
        optimizer = optim.SGD(params=chain(model.encoder.mlp.parameters(),
                                           out_MTL_layer.parameters()), lr=lr)
    elif isinstance(model, ProtoMAMLLearner):
        optimizer = optim.SGD(params=chain(model.encoder.mlp.parameters(),
                                           model.output_layer.parameters()), lr=lr)
    else:
        optimizer = optim.SGD(params=model.encoder.mlp.parameters(), lr=lr)

    cross_entropy = nn.CrossEntropyLoss()

    model.to(device)

    episode_accs = []
    for episode in episodes:
        model.train()
        # setup output layer for ProtoMAML and MTL
        if isinstance(model, ProtoMAMLLearner):
            proto_embeddings = model.proto_net(episode[0].to(device), attention_mask=episode[2].to(device))
            prototypes = model.proto_net.calculate_centroids((proto_embeddings, episode[1]), test_task.num_classes)
            W, b = model.calculate_output_params(prototypes.detach())
            model.initialize_classifier(W, b)
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
                batches_tested += 1
                if batches_tested == num_test_batches:
                    break
        episode_accs.append(np.asarray(accuracies).mean())

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

    mean, stddev = k_shot_testing(model, episodes, task, device, args.num_updates, args.num_test_batches, lr=args.lr)
    print("Mean accuracy: {}, standard deviation: {}".format(mean, stddev))

import os
import pickle

from transformers import BertTokenizer, AdamW

from util import get_test_args, get_pytorch_device, load_model
from tasks import *
from models import MetaLearner, PrototypeLearner, ProtoMAMLLearner

import torch.optim as optim
import torch.nn as nn


def k_shot_testing(model, episodes, test_iter, device, num_classes=2, num_updates=5, lr=5e-5):
    # Define optimizers and loss function
    optimizer = AdamW(params=model.parameters(), lr=lr)
    cross_entropy = nn.CrossEntropyLoss()

    for episode in episodes:
        episode_accs = []

        # setup output layer
        proto_embeddings = model.proto_net(episode[0].to(device), attention_mask=episode[2].to(device))
        prototypes = model.proto_net.calculate_centroids((proto_embeddings, episode[1]), num_classes)
        W, b = model.calculate_output_params(prototypes.detach())
        model.initialize_classifier(W, b)

        # fine-tune model with some updates on the provided episode
        for update in range(num_updates):
            optimizer.zero_grad()
            predictions = model(episode[0].to(device), attention_mask=episode[2].to(device))
            loss = cross_entropy(predictions, episode[1].long().squeeze().to(device))
            loss.backward()
            optimizer.step()

        # evaluate accuracy on whole test set
        with torch.no_grad():
            accuracies = []
            for batch in test_iter:
                predictions = model(batch[0].to(device), attention_mask=batch[2].to(device))
                acc = accuracy_score(batch[1], predictions.argmax(dim=1).cpu())
                accuracies.append(acc)
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

    print("Loading model from snapshot")
    if args.model == 'MTL':
        model = MetaLearner(args)
    elif args.model == 'ProtoNet':
        model = PrototypeLearner(args)
    elif args.model == 'ProtoMAML':
        model = ProtoMAMLLearner(args)
    else:
        RuntimeError('Unknown model type!')
    model = load_model(args.resume_snapshot, model, args.unfreeze_num, device)
    model.to(device)

    # TODO replace with options of test datasets
    if args.task == 'OffenseEval':
        task = OffensevalTask()
    else:
        RuntimeError('Unknown evaluation task!')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if not args.episodes == '':
        episodes = pickle.load(open(args.episodes, "rb"))
    else:
        episodes = sample_episodes(args.k, task, tokenizer, args.generate_episodes)
        random_id = int(np.random.randint(0, 10000, 1))
        pickle.dump(episodes, open(args.save_path+"/episodes_{}.pkl".format(random_id), "wb"))

    test_iter = task.get_iter('test', tokenizer, shuffle=False)

    mean, stddev = k_shot_testing(model, episodes, test_iter, device, task.num_classes, args.num_updates, lr=args.lr)
    print("Mean accuracy: {}, standard deviation: {}".format(mean, stddev))

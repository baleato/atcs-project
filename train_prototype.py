import os
import time
import sys
import glob
from datetime import timedelta
from itertools import chain

from torch import load
import torch.nn as nn
import torch
from transformers import BertTokenizer, AdamW, get_cosine_schedule_with_warmup

from util import get_args, get_pytorch_device
from k_shot_testing import k_shot_testing
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import PrototypeLearner

from datetime import datetime
import torch.optim as optim



def train(tasks, model, args, device):
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":", "_")))

    header = '      Time                     Task  Iteration   Progress  %Epoch       ' + \
             'Loss   Dev/Loss     Accuracy      Dev/Acc'
    log_template = '{:>10} {:>25} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}% ' + \
                   '{:10.6f}              {:10.6f}'
    test_template = 'Test mean: {}, Test std: {}'

    print(header)
    start = time.time()

    # Define optimizers and loss function

    optimizer_bert = AdamW(params=model.encoder.bert.parameters(), lr=args.bert_lr)
    optimizer = optim.Adam(params=chain(model.encoder.mlp.parameters()),
                           lr=args.lr)
    scheduler_bert = get_cosine_schedule_with_warmup(optimizer_bert, 200, args.num_iterations)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.num_iterations)
    criterion = nn.CrossEntropyLoss()

    # TODO maybe find nicer solution for passing(handling) the tokenizer
    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # initialize task sampler

    sampler = TaskSampler(tasks, method='random', supp_query_split=True)

    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
    train_iter_len = len(train_iter)
    model.train()
    model.to(device)

    # setup test model, task and episodes for evaluation
    test_model = type(model)(args)
    test_task = SentimentAnalysis(cls_dim=args.mlp_dims[-1])
    episodes = torch.load(args.episodes)

    iterations = 0
    iterations, running_loss = 0, 0.0
    best_test_mean = -1
    best_test_last = -1
    convergence_tolerance_cnt = 0
    for i in range(args.num_iterations):

        # Iterate over the data
        batch = next(train_iter)
        # prepare support and query set
        support_tuple = batch[:3]
        query_tuple = batch[3:]
        support = support_tuple[0].to(device)
        support_labels = support_tuple[1]
        support_mask = support_tuple[2].to(device)
        query = query_tuple[0].to(device)
        query_labels = query_tuple[1]
        query_mask = query_tuple[2].to(device)

        # Reset .grad attributes for weights
        optimizer_bert.zero_grad()
        optimizer.zero_grad()


        # Feed sentences into BERT instance, compute loss, perform backward
        # pass, update weights.

        support_embedding = model(support, attention_mask=support_mask)

        centroids = model.calculate_centroids((support_embedding, support_labels), sampler.get_task(train_iter.get_task_index()).num_classes)#, query_labels, train_iter, task)

        query_embedding = model(query, attention_mask=query_mask)
        distances = model.compute_distance(query_embedding, centroids)

        #predictions = torch.nn.functional.softmax(-distances, dim=1).argmax(dim=1)  # according to equation 2 in the paper



        loss = criterion(-distances, query_labels.squeeze(-1).long().to(device))
        if torch.isnan(loss).item():
            print(centroids[0])
            raise Exception("Got NaNs in loss function. Happens only sometimes... Investigate why!")
        loss.backward()
        optimizer.step()
        optimizer_bert.step()
        scheduler.step()
        scheduler_bert.step()

        running_loss += loss.item()
        iterations += 1
        if iterations % args.log_every == 0:
            acc = sampler.calculate_accuracy(-distances, query_labels.to(device))
            iter_loss = running_loss / args.log_every
            writer.add_scalar('{}/Accuracy/train'.format(sampler.get_name()), acc, iterations)
            writer.add_scalar('{}/Loss/train'.format(sampler.get_name()), iter_loss, iterations)
            print(log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                sampler.get_name(),
                iterations,
                i + 1, train_iter_len,
                (i + 1) / train_iter_len * 100,
                iter_loss, acc))
            running_loss = 0.0

        # evaluate in k shot fashion
        if iterations % args.eval_every == 0:
            test_model.load_state_dict(model.state_dict())
            test_mean, test_std = k_shot_testing(test_model, episodes, test_task, device, num_test_batches=args.num_test_batches)
            writer.add_scalar('{}/Acc'.format(test_task.get_name()), test_mean, iterations)
            writer.add_scalar('{}/STD'.format(test_task.get_name()), test_std, iterations)
            print(test_template.format(test_mean, test_std), flush=True)
            if test_mean > best_test_mean:
                best_test_mean = test_mean
                snapshot_prefix = os.path.join(args.save_path, 'best_test_{}'.format(test_task.get_name()))
                snapshot_path = (
                        snapshot_prefix +
                        '_acc_{:.5f}_iter_{}_model.pt'
                ).format(best_test_mean, iterations)
                model.save_model(snapshot_path)
                # Keep only the best snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            if test_mean > best_test_last:
                best_test_last = best_test_mean
                convergence_tolerance_cnt = 0
            else:
                convergence_tolerance_cnt += 1

            if convergence_tolerance_cnt == args.convergence_tolerance:
                break

        # saving redundant parameters
        # Save model checkpoints.
        if iterations % args.save_every == 0:
            acc = sampler.calculate_accuracy(-distances, query_labels.to(device))
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = (
                    snapshot_prefix +
                    '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
            ).format(acc, loss.item(), iterations)

            model.save_model(snapshot_path)
            # Keep only the last snapshot
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

    writer.close()


if __name__ == '__main__':
    args = get_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)
    print(device)

    if args.resume_snapshot:
        print("Loading models from snapshot")
        model = PrototypeLearner(args)
        print("Tasks")
        tasks = []
        for emotion in SemEval18SingleEmotionTask.EMOTIONS:
            tasks.append(SemEval18SingleEmotionTask(emotion))
        tasks.append(SarcasmDetection())
        tasks.append(OffensevalTask())
        model.load_model(args.resume_snapshot, device)
    else:
        print("Tasks")
        tasks = []
        for emotion in SemEval18SingleEmotionTask.EMOTIONS:
            tasks.append(SemEval18SingleEmotionTask(emotion, cls_dim=args.mlp_dims[-1]))
        tasks.append(SarcasmDetection(cls_dim=args.mlp_dims[-1]))
        tasks.append(OffensevalTask(cls_dim=args.mlp_dims[-1]))

        model = PrototypeLearner(args)
    model.to(device)
    results = train(tasks, model, args, device)

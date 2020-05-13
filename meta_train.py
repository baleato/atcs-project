import os
import time
import sys
import glob
from datetime import timedelta

from torch import load
import torch.nn as nn
import torch
from transformers import BertTokenizer, AdamW

from util import (
    get_args, get_pytorch_device, get_model, load_model,
    save_model, split_episode, compute_prototypes, initiallize_classifier)
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import MetaLearner

from datetime import datetime
import torch.optim as optim


def meta_train(tasks, method='random', custom_task_ratio=None, meta_iters=1000, num_updates=1, num_samples=4, meta_batch_size=1):
    """
    We'll start with binary classifiers (2-way classification)
    for step in range(num_steps):
        # Training
        for i in num_samples:
            task_batch_train := Sample tasks based on meta_batch_size (training set) (and task frequencies)
            for task in task_batch_train:
                forward
                loss
                backward

        # Meta-training
        if step % meta_every == 0:
            task_batch_test :=  Sample tasks not included in task_batch_train
                                meta_batch_test_size (> meta_batch_size, all?)
            for task in task_batch_test:
                forward
                loss
                backward

    params:
        - tasks
        - method: method of the task sampling sequential, custom probabilities or proportional to sqrt of data size
        - custom_task_ratio: default None only pass if custom task probabilities as sampling method
        - meta_iters: number of meta-training iterations
        - num_updates: number of updates in inner loop on same task_batch
        - num_classes: number of classes (N in N-way classification.). Default 2.
        - num_samples: examples for inner gradient update (K in K-shotlearning).
        - meta_batch_size: number of N-way tasks per batch
    """
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":", "_")))

    header = '      Time      Task      Iteration      Loss   Dev/Loss     Accuracy      Dev/Acc'
    log_template = '{:>10} {:>20} {:10.0f} {:10.6f}              {:10.6f}'

    print(header)
    start = time.time()

    # Define optimizers and loss function
    # TODO validate if BertAdam works better and then also use in MTL training
    optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=False)

    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('done.')

    sampler = TaskSampler(tasks, method=method, custom_task_ratio=custom_task_ratio)

    best_dev_acc = -1
    iterations, running_loss = 0, 0.0
    # Iterate over the data
    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
    model.train()
    # outer loop (meta-iterations)
    for i in range(meta_iters):
        grads = []
        # inner loop (sample different tasks)
        for task_sample in range(meta_batch_size):
            # clone original model
            task_model = type(model)()
            task_model.load_state_dict(model.state_dict())
            task_model.train()

            # new optimizer for every new task model
            task_optimizer = AdamW(params=task_model.parameters(), lr=args.lr, correct_bias=False)

            # prepare support and query set
            batch = next(train_iter)
            support, query = split_episode(batch)

            # setup output layer (via prototype network)
            # TODO ensure functionality
            prototypes = compute_prototypes(model, sampler.get_name(), support)
            initiallize_classifier(task_model, prototypes.detach())

            # train some iterations on support set
            for update in range(num_updates):
                task_optimizer.zero_grad()
                predictions = task_model(support[0].to(device), sampler.get_name(), attention_mask=support[2].to(device))
                task_loss = sampler.get_loss(predictions, support[1].to(device))
                task_loss.backward()
                task_optimizer.step()

            # trick to add prototypes back to computation graph
            W = prototypes + W.detach() - prototypes.detach()
            b = prototypes + b.detach() - prototypes.detach()
            initiallize_classifier(task_model, W, b)

            # calculate gradients for meta update on the query set
            predictions = task_model(query[0].to(device), sampler.get_name(), attention_mask=query[2].to(device))
            query_loss = sampler.get_loss(predictions, query[1].to(device))
            query_loss.backward()

            # save gradients of first task model
            if task_sample == 0:
                for param in task_model.parameters():
                    grads.append(param.grad)
            # add the gradients of all task samples
            else:
                for p, param in enumerate(task_model.parameters()):
                    grads[p] += param.grad

        # perform meta update
        # first load the calculated gradients in the meta-model
        for p, param in enumerate(model.parameters()):
            param.grad = grads[p]
        # update model parameters according to the gradients from inner loop
        optimizer.step()

        running_loss += query_loss.item()
        iterations += 1
        if iterations % args.log_every == 0:
            acc = task.calculate_accuracy(predictions, query[1].to(device))
            iter_loss = running_loss / args.log_every
            writer.add_scalar('{}/Accuracy/train'.format(task.get_name()), acc, iterations)
            writer.add_scalar('{}/Loss/train'.format(task.get_name()), iter_loss, iterations)
            print(log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                task.get_name(),
                iterations,
                iter_loss, acc))
            running_loss = 0.0

        # saving redundant parameters
        # Save model checkpoints.
        if iterations % args.save_every == 0:
            acc = task.calculate_accuracy(predictions, query[1].to(device))
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = (
                    snapshot_prefix +
                    '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
            ).format(acc, query_loss.item(), iterations)
            # FIXME: save_model
            # save_model(model, args.unfreeze_num, snapshot_path)
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

    if args.resume_snapshot:
        print("Loading models from snapshot")
        model = MetaLearner(args)
        model = load_model(args.resume_snapshot, model, args.unfreeze_num, device)
    else:
        model = MetaLearner(args)
        model.to(device)
        print("Tasks")
        tasks = []
        # tasks.append(SemEval18Task())
        tasks.append(SemEval18SurpriseTask())
        tasks.append(SemEval18TrustTask())
        tasks.append(SarcasmDetection())
        tasks.append(OffensevalTask())
        for task in tasks:
            model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
        sampler = TaskSampler(tasks, method='random')
    results = train([sampler], model, args, device)

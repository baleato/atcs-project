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
    save_model, split_episode)
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
    dev_log_template = '{:>10} {:>20} {:10.0f} {:10.6f}              {:12.6f}'

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
    # outer loop
    for i in range(meta_iters):
        task_models = []
        # inner loop (sample different tasks)
        for task_sample in range(meta_batch_size):
            task_model = model.clone()
            task_optimizer = AdamW(params=task_model.parameters(), lr=args.lr, correct_bias=False)
            batch = next(train_iter)

            # save task speciffic models for meta update
            task_models.append({'model': task_model, 'task': train_iter.get_task_index()})

            for update in range(num_updates):
                support, query = split_episode(batch)
                # TODO ensure functionality
                # prototypes = model.compute_prototypes(support)
                # model.initiallize_classifier(prototypes)
                task_optimizer.zero_grad()
                predictions = task_model(query[0].to(device), sampler.get_name(), attention_mask=query[2].to(device))

                task_loss = sampler.get_loss(predictions, query[1].to(device))
                task_loss.backward()
                task_optimizer.step()

        meta_losses = []
        for task_sample in task_models:
            # get a new sample of the same task for meta training
            batch = train_iter.get_task_batch(task_sample['task'])
            task = sampler.get_task(task_sample['task'])

            support, query = split_episode(batch, ratio=0.5)
            # prototypes = model.compute_prototypes(support)
            # model.initiallize_classifier(prototypes)

            sentences = query[0].to(device)
            labels = query[1]
            attention_masks = query[2].to(device)

            predictions = model(sentences, task.get_name(), attention_mask=attention_masks)

            meta_losses.append(task.get_loss(predictions, labels.to(device)))
        meta_loss = sum(meta_losses)
        meta_loss.backward()
        optimizer.step()

        running_loss += meta_loss.item()
        iterations += 1
        if iterations % args.log_every == 0:
            acc = task.calculate_accuracy(predictions, labels.to(device))
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
            acc = task.calculate_accuracy(predictions, labels.to(device))
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = (
                    snapshot_prefix +
                    '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
            ).format(acc, loss.item(), iterations)
            # FIXME: save_model
            # save_model(model, args.unfreeze_num, snapshot_path)
            # Keep only the last snapshot
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # ============================ EVALUATION ============================
        dev_iter = sampler.get_iter('dev', tokenizer, batch_size=args.batch_size)
        dev_iter_len = len(dev_iter)
        model.eval()

        # calculate accuracy on validation set
        sum_dev_loss, sum_dev_acc = 0, 0
        with torch.no_grad():
            for dev_batch in dev_iter:
                sentences = dev_batch[0].to(device)
                labels = dev_batch[1]
                attention_masks = dev_batch[2].to(device)
                outputs = model(sentences, task.get_name(), attention_mask=attention_masks)
                # Loss
                batch_dev_loss = task.get_loss(outputs, labels.to(device))
                sum_dev_loss += batch_dev_loss.item()
                # Accuracy
                acc = task.calculate_accuracy(outputs, labels.to(device))
                sum_dev_acc += acc
        dev_acc = sum_dev_acc / dev_iter_len
        dev_loss = sum_dev_loss / dev_iter_len

        print(dev_log_template.format(
            str(timedelta(seconds=int(time.time() - start))),
            task.get_name(),
            iterations,
            dev_loss, dev_acc))

        writer.add_scalar('{}/Accuracy/dev'.format(task.get_name()), dev_acc, iterations)
        writer.add_scalar('{}/Loss/dev'.format(task.get_name()), dev_loss, iterations)

        if best_dev_acc < dev_acc:
            best_dev_acc = dev_acc
            snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
            snapshot_path = (
                    snapshot_prefix +
                    '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
            ).format(dev_acc, dev_loss, iterations)
            # FIXME: save_model
            # save_model(model, args.unfreeze_num, snapshot_path)
            # Keep only the best snapshot
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

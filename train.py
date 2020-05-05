import os
import time
import sys
import glob
from datetime import timedelta

from torch import load
import torch.nn as nn
import torch
from transformers import BertTokenizer

from util import (
    get_args, get_pytorch_device, create_iters, get_model, load_model,
    save_model)
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import MetaLearner

from datetime import datetime
import torch.optim as optim


def meta_train():
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
        - num_classes: number of classes (N in N-way classification.). Default 2.
        - num_samples: examples for inner gradient update (K in K-shotlearning).
        - meta_batch_size: number of N-way tasks per batch
    """
    pass


def train(tasks, model, args, device):
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":","_")))
    header = '      Time                 Task   Epoch  Iteration   Progress  %Epoch       ' + \
        'Loss   Dev/Loss     Accuracy      Dev/Acc'
    log_template = '{:>10} {:>20} {:7.0f} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}% ' + \
        '{:10.6f}              {:10.6f}'
    dev_log_template = '{:>10} {:>20} {:7.0f} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}%' + \
        '            {:10.6f}              {:12.6f}'
    print(header)
    start = time.time()

    # Define optimizers and loss function
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    best_dev_acc = -1
    iterations, running_loss = 0, 0.0
    for epoch in range(args.max_epochs):
        for task in tasks:
            # Iterate over the data
            train_iter = task.get_iter('train', batch_size=args.batch_size, shuffle=True)
            train_iter_len = task.get_num_batches('train', batch_size=args.batch_size)
            dev_iter_len = task.get_num_batches('dev', batch_size=args.batch_size)
            model.train()
            for batch_idx, batch in enumerate(train_iter):
                # Reset .grad attributes for weights
                optimizer.zero_grad()

                # Extract the sentence_ids and target vector, send sentences to GPU
                sentences = batch[0].to(device)
                labels = batch[1]
                attention_masks = batch[2].to(device)

                # Feed sentences into BERT instance, compute loss, perform backward
                # pass, update weights.
                predictions = model(sentences, task.NAME, attention_mask=attention_masks)

                loss = task.get_loss(predictions, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                iterations += 1
                if iterations % args.log_every == 0:
                    acc = task.calculate_accuracy(predictions, labels)
                    iter_loss = running_loss / args.log_every
                    writer.add_scalar('{}/Accuracy/train'.format(task.NAME), acc, iterations)
                    writer.add_scalar('{}/Loss/train'.format(task.NAME), iter_loss, iterations)
                    print(log_template.format(
                        str(timedelta(seconds=int(time.time() - start))),
                        task.NAME,
                        epoch,
                        iterations,
                        batch_idx+1, train_iter_len,
                        (batch_idx+1) / train_iter_len * 100,
                        iter_loss, acc))
                    running_loss = 0.0

                # saving redundant parameters
                # Save model checkpoints.
                if iterations % args.save_every == 0:
                    acc = task.calculate_accuracy(predictions, labels)
                    snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                    snapshot_path = (
                            snapshot_prefix +
                            '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
                        ).format(acc, loss.item(), iterations)
                    save_model(model, snapshot_path)
                    # Keep only the last snapshot
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)

            # ============================ EVALUATION ============================
            model.eval()

            # calculate accuracy on validation set
            sum_dev_loss, sum_dev_acc = 0, 0
            with torch.no_grad():
                dev_iter = task.get_iter('dev',
                                                       batch_size=args.batch_size)
                for dev_batch in dev_iter:
                    sentences = dev_batch[0].to(device)
                    labels = dev_batch[1]
                    outputs = model(sentences, task.NAME)
                    # Loss
                    batch_dev_loss = task.get_loss(outputs, labels)
                    sum_dev_loss += batch_dev_loss.item()
                    # Accuracy
                    acc = task.calculate_accuracy(outputs, labels)
                    sum_dev_acc += acc
            dev_acc = sum_dev_acc / dev_iter_len
            dev_loss = sum_dev_loss / dev_iter_len

            print(dev_log_template.format(
                    str(timedelta(seconds=int(time.time() - start))),
                    task.NAME,
                    epoch,
                    iterations,
                    batch_idx+1, train_iter_len,
                    (batch_idx+1) / train_iter_len * 100,
                    dev_loss, dev_acc))

            writer.add_scalar('{}/Accuracy/dev'.format(task.NAME), dev_acc, iterations)
            writer.add_scalar('{}/Loss/dev'.format(task.NAME), dev_loss, iterations)

            if best_dev_acc < dev_acc:
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = (
                        snapshot_prefix +
                        '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
                    ).format(dev_acc, dev_loss, iterations)
                save_model(model, snapshot_path)
                # Keep only the best snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

    writer.close()

print('Loading Tokenizer..')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# TODO: move tokenizer to tasks and always assume BERT for symplicity
def fn_tokenizer(sentences, max_length=32):
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'  # returns results already as pytorch tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Stack the input_ids, labels and attention_masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


if __name__ == '__main__':
    args = get_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)

    if args.resume_snapshot:
        print("Loading models from snapshot")
        model = load_model(args.resume_snapshot, device)
    else:
        model = MetaLearner(args)
        model.to(device)
        print("Tasks")
        tasks = []
        # tasks.append(SemEval18Task(fn_tokenizer=fn_tokenizer))
        tasks.append(SemEval18SurpriseTask(fn_tokenizer=fn_tokenizer))
        tasks.append(SemEval18TrustTask(fn_tokenizer=fn_tokenizer))
        for task in tasks:
            model.add_task_classifier(task.NAME, task.get_classifier())
    results = train(tasks, model, args, device)

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
    save_model)
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import MetaLearner, PrototypeLearner

from datetime import datetime
import torch.optim as optim


def train(tasks, model, args, device):
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":", "_")))

    header = '      Time                     Task   Epoch  Iteration   Progress  %Epoch       ' + \
        'Loss   Dev/Loss     Accuracy      Dev/Acc'
    log_template = '{:>10} {:>25} {:7.0f} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}% ' + \
        '{:10.6f}              {:10.6f}'
    dev_log_template = '{:>10} {:>25} {:7.0f} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}%' + \
        '            {:10.6f}              {:12.6f}'

    print(header)
    start = time.time()

    # Define optimizers and loss function
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # TODO maybe find nicer solution for passing(handling) the tokenizer
    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    best_dev_acc = -1
    iterations, running_loss = 0, 0.0
    for epoch in range(args.max_epochs):
        for task in tasks:
            # Iterate over the data
            train_iter = task.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
            train_iter_len = len(train_iter)
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
                predictions = model(sentences, task.get_name(), attention_mask=attention_masks)

                loss = task.get_loss(predictions, labels.to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                iterations += 1
                if iterations % args.log_every == 0:
                    acc = task.calculate_accuracy(predictions, labels.to(device))
                    iter_loss = running_loss / args.log_every
                    writer.add_scalar('{}/Accuracy/train'.format(task.get_name()), acc, iterations)
                    writer.add_scalar('{}/Loss/train'.format(task.get_name()), iter_loss, iterations)
                    print(log_template.format(
                        str(timedelta(seconds=int(time.time() - start))),
                        task.get_name(),
                        epoch,
                        iterations,
                        batch_idx+1, train_iter_len,
                        (batch_idx+1) / train_iter_len * 100,
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
            dev_iter = task.get_iter('dev', tokenizer, batch_size=args.batch_size)
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
                    epoch,
                    iterations,
                    batch_idx+1, train_iter_len,
                    (batch_idx+1) / train_iter_len * 100,
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
        for emotion in SemEval18SingleEmotionTask.EMOTIONS:
            tasks.append(SemEval18SingleEmotionTask(emotion))
        tasks.append(SarcasmDetection())
        tasks.append(OffensevalTask())
        for task in tasks:
            model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
        sampler = TaskSampler(tasks, method='random')

    results = train([sampler], model, args, device)

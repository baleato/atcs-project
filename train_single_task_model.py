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
from models import MultiTaskLearner

from datetime import datetime
import torch.optim as optim

def test(device, model, task, task_iter):
    dev_task_accs = []
    dev_iter = task_iter
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
    return dev_loss, dev_acc

def train(task, model, args, device):
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":", "_")))

    header = '      Time                     Task      Epoch   Progress  %Epoch       ' + \
        'Loss   Dev/Loss     Accuracy      Dev/Acc'
    log_template = '{:>10} {:>25} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}% ' + \
        '{:10.6f}              {:10.6f}'
    dev_log_template = '{:>10} {:>25} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}%' + \
        '            {:10.6f}              {:12.6f}'
    test_template = 'Test mean: {}, Test std: {}'

    print(header)
    start = time.time()

    # Define optimizers and loss function
    optimizer_bert = AdamW(params=model.encoder.bert.parameters(), lr=args.bert_lr)
    # TODO: don't access model internals, export function to get desired parameters
    task_classifiers_params = [model._modules[m_name].parameters() for m_name in model._modules if 'task' in m_name]
    optimizer = optim.Adam(params=chain(model.encoder.mlp.parameters(),
                                        *task_classifiers_params),
                           lr=args.lr)
    scheduler_bert = get_cosine_schedule_with_warmup(optimizer_bert, 200, args.num_iterations)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.num_iterations)

    # TODO maybe find nicer solution for passing(handling) the tokenizer
    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    best_dev_acc = -1
    iterations, running_loss = 0, 0.0
    best_test_mean = -1
    # TODO: decide stopping on error
    for epoch in range(1, args.max_epochs + 1):
        train_iter = task.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=False)
        train_iter_len = len(train_iter)
        model.train()
        for batch_idx, batch in enumerate(train_iter):
            # Reset .grad attributes for weights
            optimizer_bert.zero_grad()
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
            optimizer_bert.step()
            scheduler.step()
            scheduler_bert.step()

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
                    iterations, train_iter_len,
                    (batch_idx + 1) / train_iter_len * 100,
                    iter_loss, acc))
                running_loss = 0.0

            # saving redundant parameters
            # Save model checkpoints.
            if iterations % args.save_every == 0:
                acc = task.calculate_accuracy(predictions, labels.to(device))
                snapshot_prefix = os.path.join(args.save_path, 'snapshot_{}'.format(task.get_name()))
                snapshot_path = (
                        snapshot_prefix +
                        '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'
                    ).format(acc, loss.item(), iterations)
                model.save_model(snapshot_path)
                # Keep only the last snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        # ============================ EVALUATION ============================
        dev_iter = task.get_iter('dev', tokenizer, batch_size=args.batch_size)
        dev_loss, dev_acc = test(device, model, task, dev_iter)

        print(dev_log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                task.get_name(),
                epoch,
                iterations, train_iter_len,
                (batch_idx + 1) / train_iter_len * 100,
                dev_loss, dev_acc))

        writer.add_scalar('{}/Accuracy/dev'.format(task.get_name()), dev_acc, iterations)
        writer.add_scalar('{}/Loss/dev'.format(task.get_name()), dev_loss, iterations)

        if best_dev_acc < dev_acc:
            best_dev_acc = dev_acc
            snapshot_prefix = os.path.join(args.save_path, 'best_snapshot_{}'.format(task.get_name()))
            snapshot_path = (
                    snapshot_prefix +
                    '_acc_{:.4f}_iter_{}_model.pt'
                ).format(dev_acc, iterations)
            model.save_model(snapshot_path)
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

    task_name = args.single_train_task
    if task_name == 'SentimentAnalysis':
        task = SentimentAnalysis(cls_dim=args.mlp_dims[-1])
    elif task_name == 'IronySubtaskA':
        task = IronySubtaskA(cls_dim=args.mlp_dims[-1])
    elif task_name == 'IronySubtaskB':
        task = IronySubtaskB(cls_dim=args.mlp_dims[-1])
    elif task_name == 'Abuse':
        task = Abuse(cls_dim=args.mlp_dims[-1])
    elif task_name == 'Politeness':
        task = Politeness(cls_dim=args.mlp_dims[-1])

    if args.resume_snapshot:
        print("Loading models from snapshot")
        model = MultiTaskLearner(args)
        model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
        model.load_model(args.resume_snapshot, device)
    else:

        model = MultiTaskLearner(args)
        model.to(device)
        model.add_task_classifier(task.get_name(), task.get_classifier().to(device))

    train(task, model, args, device)

    # TEST
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot_{}'.format(task.get_name()))
    for f in glob.glob(snapshot_prefix + '*'):
        best_model = MultiTaskLearner(args)
        best_model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
        best_model.load_model(f, device)
        test_iter = task.get_iter('test', tokenizer, batch_size=args.batch_size)
        test_loss, test_acc = test(device, model, task, test_iter)
        print('Model:', f)
        print('Test loss:', test_loss)
        print('Test Accuracy:',test_acc)

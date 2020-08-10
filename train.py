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


def train(tasks, model, args, device):
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":", "_")))

    header = '      Time                     Task  Iteration   Progress  %Epoch       ' + \
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

    # initialize task sampler
    sampler = TaskSampler(tasks, method='random')

    # Iterate over the data
    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
    train_iter_len = len(train_iter)
    model.train()

    # setup test model, task and episodes for evaluation
    test_task = SentimentAnalysis(cls_dim=args.mlp_dims[-1])
    test_model = type(model)(args)
    test_model.add_task_classifier(test_task.get_name(), test_task.get_classifier().to(device))
    output_layer_name = 'task_{}'.format(test_task.get_name())
    output_layer_init = test_model._modules[output_layer_name].state_dict()
    episodes = torch.load(args.episodes)

    best_dev_acc = -1
    iterations, running_loss = 0, 0.0
    best_test_mean = -1
    best_test_last = -1
    convergence_tolerance_cnt = 0
    for i in range(args.num_iterations):

        batch = next(train_iter)

        # Reset .grad attributes for weights
        optimizer_bert.zero_grad()
        optimizer.zero_grad()

        # Extract the sentence_ids and target vector, send sentences to GPU
        sentences = batch[0].to(device)
        labels = batch[1]
        attention_masks = batch[2].to(device)

        # Feed sentences into BERT instance, compute loss, perform backward
        # pass, update weights.
        predictions = model(sentences, sampler.get_name(), attention_mask=attention_masks)

        loss = sampler.get_loss(predictions, labels.to(device))
        loss.backward()
        optimizer.step()
        optimizer_bert.step()
        scheduler.step()
        scheduler_bert.step()

        running_loss += loss.item()
        iterations += 1
        if iterations % args.log_every == 0:
            acc = sampler.calculate_accuracy(predictions, labels.to(device))
            iter_loss = running_loss / args.log_every
            writer.add_scalar('{}/Accuracy/train'.format(sampler.get_name()), acc, iterations)
            writer.add_scalar('{}/Loss/train'.format(sampler.get_name()), iter_loss, iterations)
            print(log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                sampler.get_name(),
                iterations,
                i+1, train_iter_len,
                (i+1) / train_iter_len * 100,
                iter_loss, acc))
            running_loss = 0.0

        # saving redundant parameters
        # Save model checkpoints.
        if iterations % args.save_every == 0:
            acc = sampler.calculate_accuracy(predictions, labels.to(device))
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

        if iterations % args.eval_every == 0:
            # ============================ EVALUATION ============================
            dev_task_accs = []
            for task in sampler.tasks:
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
                        iterations,
                        i+1, train_iter_len,
                        (i+1) / train_iter_len * 100,
                        dev_loss, dev_acc))

                writer.add_scalar('{}/Accuracy/dev'.format(task.get_name()), dev_acc, iterations)
                writer.add_scalar('{}/Loss/dev'.format(task.get_name()), dev_loss, iterations)
                dev_task_accs.append(dev_acc)

            mean_dev_acc = sum(dev_task_accs) / len(dev_task_accs)

            if best_dev_acc < mean_dev_acc:
                best_dev_acc = mean_dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = (
                        snapshot_prefix +
                        '_acc_{:.4f}_iter_{}_model.pt'
                    ).format(mean_dev_acc, iterations)
                model.save_model(snapshot_path)
                # Keep only the best snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate in k shot fashion
            test_model.encoder.load_state_dict(model.encoder.state_dict())
            # ensure same output layer init for comparability
            test_model._modules[output_layer_name].load_state_dict(output_layer_init)
            test_mean, test_std = k_shot_testing(test_model, episodes, test_task, device,
                                          num_test_batches=args.num_test_batches)
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

    writer.close()


if __name__ == '__main__':
    args = get_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)

    if args.resume_snapshot:
        print("Loading models from snapshot")
        model = MultiTaskLearner(args)
        print("Tasks")
        tasks = []
        for emotion in SemEval18SingleEmotionTask.EMOTIONS:
            tasks.append(SemEval18SingleEmotionTask(emotion, cls_dim=args.mlp_dims[-1]))
        tasks.append(SarcasmDetection(cls_dim=args.mlp_dims[-1]))
        tasks.append(OffensevalTask(cls_dim=args.mlp_dims[-1]))
        for task in tasks:
            model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
        model.load_model(args.resume_snapshot, device)
    else:

        model = MultiTaskLearner(args)
        model.to(device)
        print("Tasks")
        tasks = []
        for emotion in SemEval18SingleEmotionTask.EMOTIONS:
            tasks.append(SemEval18SingleEmotionTask(emotion, cls_dim=args.mlp_dims[-1]))
        tasks.append(SarcasmDetection(cls_dim=args.mlp_dims[-1]))
        tasks.append(OffensevalTask(cls_dim=args.mlp_dims[-1]))
        for task in tasks:
            model.add_task_classifier(task.get_name(), task.get_classifier().to(device))

    results = train(tasks, model, args, device)

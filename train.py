import os
import time
import sys
import glob
from datetime import timedelta

from sklearn.metrics import jaccard_score
from torch import load
import torch.nn as nn
import torch

from util import (
    get_args, get_pytorch_device, create_iters, get_model, load_model,
    save_model)
from torch.utils.tensorboard import SummaryWriter
from models import MetaLearner

from datetime import datetime
import torch.optim as optim


def evaluate_emo(outputs, gold_labels):
    threshold = 0.5
    pred_labels = (outputs.clone().detach() > threshold).type_as(gold_labels)
    micro = jaccard_score(pred_labels, gold_labels, average='micro')
    macro = jaccard_score(pred_labels, gold_labels, average='macro')
    return micro, macro


def train(model, args, device):
    print("Creating DataLoaders")
    train_iter = create_iters(path='./data/semeval18_task1_class/train.txt',
                              order='random',
                              batch_size=args.batch_size)
    dev_iter = create_iters(path='./data/semeval18_task1_class/dev.txt',
                            order='random',
                            batch_size=args.batch_size)

    # Define optimizers and loss function
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now())))
    header = '      Time   Epoch  Iteration   Progress  %Epoch       ' + \
        'Loss   Dev/Loss      Micro    Dev/Micro      Macro    Dev/Macro'
    log_template = '{:>10} {:7.0f} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f}% ' + \
        '{:10.6f}            {:10.6f}              {:10.6f}'
    dev_log_template = '{:>10} {:7.0f} {:10.0f} {:5.0f}/{:<5.0f} {:6.0f}' + \
        '            {:10.6f}            {:12.6f}            {:12.6f}'
    print(header)
    start = time.time()

    # Iterate over the data
    best_dev_micro = -1
    iterations, running_loss = 0, 0.0
    for epoch in range(args.max_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_iter):
            # Reset .grad attributes for weights
            optimizer.zero_grad()

            # Extract the sentence_ids and target vector, send sentences to GPU
            sentences = batch[0].to(device)
            labels = batch[1]

            # Feed sentences into BERT instance, compute loss, perform backward
            # pass, update weights.
            predictions = model(sentences)

            loss = criterion(predictions, labels.type_as(predictions))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iterations += 1
            if iterations % args.log_every == 0:
                emo_micro, emo_macro = evaluate_emo(predictions, labels)
                iter_loss = running_loss / args.log_every
                writer.add_scalar('training loss', iter_loss, iterations)
                writer.add_scalar('training micro', emo_micro, iterations)
                writer.add_scalar('training macro', emo_macro, iterations)
                print(log_template.format(
                    str(timedelta(seconds=int(time.time() - start))),
                    epoch,
                    iterations,
                    batch_idx+1, len(train_iter),
                    (batch_idx+1) / len(train_iter) * 100,
                    iter_loss, emo_micro, emo_macro))
                running_loss = 0.0

            # saving redundant parameters
            # Save model checkpoints.
            if iterations % args.save_every == 0:
                emo_micro, emo_macro = evaluate_emo(predictions, labels)
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + \
                    '_micro_{:.4f}_macro_{:.4f}_loss_{:.6f}_iter_{}_model.pt' \
                    .format(emo_micro, emo_macro, loss.item(), iterations)
                save_model(model, snapshot_path)
                # Keep only the last snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        # ============================ EVALUATION ============================
        model.eval()

        # calculate accuracy on validation set
        sum_dev_loss, sum_micro, sum_macro = 0, 0, 0
        with torch.no_grad():
            for dev_batch in dev_iter:
                sentences = dev_batch[0].to(device)
                labels = dev_batch[1]
                outputs = model(sentences)
                # Loss
                batch_dev_loss = criterion(outputs, labels.type_as(outputs))
                sum_dev_loss += batch_dev_loss.item()
                # Accuracy
                emo_micro, emo_macro = evaluate_emo(outputs, labels)
                sum_micro += emo_micro
                sum_macro += emo_macro
        dev_loss = sum_dev_loss / len(dev_iter)
        dev_micro = sum_micro / len(dev_iter)
        dev_macro = sum_macro / len(dev_iter)

        print(dev_log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                epoch,
                iterations,
                batch_idx+1, len(train_iter),
                (batch_idx+1) / len(train_iter) * 100,
                dev_loss, dev_micro, dev_macro))

        writer.add_scalar('dev loss', dev_loss, iterations)
        writer.add_scalar('dev micro', dev_micro, iterations)
        writer.add_scalar('dev macro', dev_macro, iterations)

        if best_dev_micro < dev_micro:
            snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
            snapshot_path = snapshot_prefix + \
                '_micro_{:.4f}_macro_{:.4f}_loss_{:.6f}_iter_{}_model.pt' \
                .format(dev_micro, dev_macro, dev_loss, iterations)
            save_model(model, snapshot_path)
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
        model = load_model(args.resume_snapshot, device)
    else:
        model = MetaLearner(args)
        model.to(device)

    results = train(model, args, device)

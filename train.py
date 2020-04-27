import os
import time
import sys
import glob

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


def train(model, args):
    print("Creating DataLoaders")
    train_iter = create_iters(path='./data/semeval18_task1_class/train.txt',
                              order='random',
                              batch_size=args.batch_size)

    # Define optimizers and loss function
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now())))
    header = '      Loss      Micro      Macro'
    log_template = '{:10.6f} {:10.6f} {:10.6f}'
    print(header)

    # Iterate over the data
    iterations, running_loss = 0, 0.0
    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_iter:
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

            # Compute accuracy
            threshold = 0.5
            pred_labels = (
                predictions.clone().detach() > threshold).type_as(labels)
            emo_micro = jaccard_score(pred_labels, labels, average='micro')
            emo_macro = jaccard_score(pred_labels, labels, average='macro')

            running_loss += loss.item()
            iterations += 1
            if iterations % args.log_every == 0:
                writer.add_scalar(
                    'training loss',
                    running_loss / args.log_every, iterations)
                running_loss = 0.0
            print(log_template.format(loss.item(), emo_micro, emo_macro))

            # saving redundant parameters
            # Save model checkpoints.
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + \
                    '_micro_{:.4f}_macro_{:.4f}_loss_{:.6f}_iter_{}_model.pt' \
                    .format(emo_micro, emo_macro, loss.item(), iterations)
                save_model(model, snapshot_path)
                # Keep only the last snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        # TODO:
        # - Evaluate model on dev set
        # - Store model with best performing on dev set
        # - Log dev set results

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

    results = train(model, args)

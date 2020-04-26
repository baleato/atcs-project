import os
import time
import sys

from torch import load
import torch.nn as nn

from util import get_args, get_pytorch_device, create_iters, load_model
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.optim as optim


def train(iter, model, args):
    # Define optimizers and loss function
    optimizer = optim.Adam(params=model.parameters(), lr=0.00002)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(os.path.join(args.save_path, 'runs', '{}'.format(datetime.now())))
    # Iterate over the data
    iterations, running_loss = 0, 0.0
    for epoch in range(10):
        epoch_loss = []
        for i, batch in enumerate(iter):

            # Reset .grad attributes for weights
            optimizer.zero_grad()

            # Extract the sentence_ids and target vector, send sentences to GPU
            sentences = batch[0].to(device)
            labels = batch[1]

            # Feed sentences into BERT instance, compute loss, perform backward pass, update weights.
            iterations += 1
            predictions = model(sentences)[0]
            labels = labels.type_as(predictions)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            running_loss += loss.item()
            if iterations % args.log_every == 0:
                writer.add_scalar('training loss', running_loss / args.log_every, iterations)
                running_loss = 0.0
            print(loss.item())
    writer.close()


if __name__ == '__main__':
    args = get_args()
    device = get_pytorch_device(args)

    print("Creating DataLoaders")
    train_iter = create_iters(path='./data/semeval18_task1_class/train.txt',
                         order='random',
                         batch_size=32)


    # TODO: Allow for resuming a previously trained model
    model = load_model()
    model = model.to(device)

    results = train(train_iter, model, args)

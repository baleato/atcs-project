import os
import time
import sys

from torch import load
import torch.nn as nn

# from model import MetaLearner
from util import get_args, get_pytorch_device, create_iters, load_model
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.optim as optim


args = get_args()
device = get_pytorch_device(args)

# # TODO: check that we can load learner with bert the same way
# if args.resume_snapshot:
#     model = load(args.resume_snapshot, map_location=device)
# else:
#     model = MetaLearner(config)
#     model.to(device)

# TODO: Load datasets splits

writer = SummaryWriter(os.path.join(args.save_path, 'runs', '{}'.format(datetime.now())))

# TODO: training loop

writer.close()

def train(iter, model):
    # Define optimizers and loss function
    optimizer = optim.Adam(params=model.parameters(), lr=0.00002)
    criterion = nn.BCEWithLogitsLoss()

    # Iterate over the data
    for epoch in range(10):
        epoch_loss = []
        for i, batch in enumerate(iter):

            # Reset .grad attributes for weights
            optimizer.zero_grad()

            # Extract the sentence_ids and target vector
            sentences = batch[0]
            labels = batch[1]

            # Feed sentences into BERT instance, compute loss, perform backward pass, update weights.
            predictions = model(sentences)[0]
            labels = labels.type_as(predictions)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            print(loss.item())

def main():
    args = get_args()
    device = get_pytorch_device(args)

    print("Creating DataLoaders")
    train_iter = create_iters(path='./data/sem_eval_2018/train.txt',
                         order='random',
                         batch_size=32)
    model = load_model()
    model = model.to(device)

    results = train(train_iter, model)

main()

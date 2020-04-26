import os
import time
import sys

from sklearn.metrics import jaccard_score
from torch import load
import torch.nn as nn
import torch

from util import get_args, get_pytorch_device, create_iters, load_model
from torch.utils.tensorboard import SummaryWriter
from models import MLPClassifier

from datetime import datetime
import torch.optim as optim

def train(iter, model, classifier, params, args):
    # Define optimizers and loss function
    optimizer = optim.Adam(params=params, lr=0.00002)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(os.path.join(args.save_path, 'runs', '{}'.format(datetime.now())))
    header = '      Loss      Micro      Macro'
    log_template = '{:10.6f} {:10.6f} {:10.6f}'
    print(header)
    # Iterate over the data
    iterations, running_loss = 0, 0.0
    for epoch in range(10):
        for batch in iter:
            # Reset .grad attributes for weights
            optimizer.zero_grad()

            # Extract the sentence_ids and target vector, send sentences to GPU
            sentences = batch[0].to(device)
            labels = batch[1]

            # Feed sentences into BERT instance, compute loss, perform backward pass, update weights.
            output = model(sentences)[0]
            predictions = classifier(output)

            loss = criterion(predictions, labels.type_as(predictions))
            loss.backward()
            optimizer.step()

            # Compute accuracy
            threshold = 0
            pred_labels = (predictions.clone().detach() > threshold).type_as(labels)
            emo_micro = jaccard_score(pred_labels, labels, average='micro')
            emo_macro = jaccard_score(pred_labels, labels, average='macro')

            running_loss += loss.item()
            iterations += 1
            if iterations % args.log_every == 0:
                writer.add_scalar('training loss', running_loss / args.log_every, iterations)
                running_loss = 0.0
            print(log_template.format(loss.item(), emo_micro, emo_macro))
    writer.close()


if __name__ == '__main__':
    args = get_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)

    print("Creating DataLoaders")
    train_iter = create_iters(path='./data/semeval18_task1_class/train.txt',
                         order='random',
                         batch_size=args.batch_size)


    # TODO: Allow for resuming a previously trained model
    # Load instance of BERT
    model = load_model()

    # Defining a custom MLP for emotion classification
    classifier = MLPClassifier(input_dim=768, target_dim=11)

    # Option to freeze BERT parameters
    if args.freeze_bert:
        print('Freezing the first {} BERT layers'.format(args.freeze_num))
        for i, param in enumerate(model.parameters()):
            param.requires_grad = False
            if i+1 == args.freeze_num:
                break

    # Define params to send to optimizer
    params = list(classifier.parameters()) + list(model.parameters())

    model = model.to(device)
    classifier = classifier.to(device)
    
    results = train(train_iter, model, classifier, params, args)

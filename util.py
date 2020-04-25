import os
from argparse import ArgumentParser
import pandas as pd
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset

from transformers import BertTokenizer
from transformers import BertForMultipleChoice, AdamW, BertConfig

import sys

def create_iters(path, order, batch_size):
    """
    Function that takes a data file as input, applies processing as required by BERT,
    maps words to IDs, returns DataLoader iterable.
    :param path: path to data file
    :param order: determines sampling order when creating DataLoaders
        if arg == 'random':
            sampler = RandomSampler   [this should be used for training set]
        if arg == 'sequential'
            sampler = SequentialSampler     [this should be used for dev/test sets]
    :param batch_size:
    :return:
    """
    # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
    data_df = pd.read_csv(path, sep='\t')
    sentences = data_df.Tweet.values
    labels = data_df[['anger', 'anticipation', 'disgust', 'fear', 'joy',
       'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']].values

    # add BERT-required formatting; tokenize with desired BertTokenizer
    # Load Tokenizer
    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    for sentence in sentences:
        sentence_ids = tokenizer.encode(
            sentence,
            add_special_tokens = True,
            max_length = 64,
            pad_to_max_length = True
        )
        input_ids.append(torch.tensor(sentence_ids))

    # Convert input_ids and labels to tensors;
    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.tensor(labels)

    # Load tensors into torch Dataset object
    dataset = TensorDataset(input_ids, labels)
    # Determine what sampling mode should be used
    if order == 'random':
        sampler = RandomSampler(dataset)
    elif order == 'sequential':
        sampler = SequentialSampler(dataset)

    # Create DataLoader object
    dataloader = DataLoader(dataset,
                            sampler = sampler,
                            batch_size = batch_size
                            )
    return dataloader

# # Here is some code for testing
# test_iter = create_iters(path='./data/sem_eval_2018/test.txt',
#                          order='sequential',
#                          batch_size=64)
# for batch in test_iter:
#     print(len(batch[0]))
#     sys.exit()

def load_model():
    """
    this function loads a pre-trained bert-base-uncased model instance with a multi-label
    classification layer on top. Running it for the first time will take about half a
    minute or so, as the parameters need to be downloaded to your machine.
    :return:
    """
    print('Loading pre-trained BERT')
    model = BertForMultipleChoice.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=11,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    return model

bertybert = load_model()

def get_pytorch_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    return device


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default=os.path.join(os.getcwd(), '.data'))
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--lr',type=float, default=.1)
    args = parser.parse_args()
    return args

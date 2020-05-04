import os
from argparse import ArgumentParser
import pandas as pd
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, \
    BertModel

import sys


def create_iters(path, order, batch_size, path2=''):
    """
    Function that takes a data file as input, applies processing as required by
    BERT, maps words to IDs, returns DataLoader iterable.
    :param path: path to data file
    :param order: determines sampling order when creating DataLoaders
        if arg == 'random':
            sampler = RandomSampler     [this should be used for training set]
        if arg == 'sequential'
            sampler = SequentialSampler [this should be used for dev/test sets]
    :param batch_size:
    :return:
    """
    if 'offenseval' in path:
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        data_df = pd.read_csv(path, sep='\t')
        sentences = data_df.tweet.values
        if 'testset' in path:
            assert path2 != '', "Missing path to gold labels!"
            data_df = pd.read_csv(path2, sep=',', header=None)
            data_df[1].replace(to_replace='OFF', value=1, inplace=True)
            data_df[1].replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df[1].values
        else:
            data_df.subtask_a.replace(to_replace='OFF', value=1, inplace=True)
            data_df.subtask_a.replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df.subtask_a.values
        max_length = 64
    else:
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        data_df = pd.read_csv(path, sep='\t')
        sentences = data_df.Tweet.values
        labels = data_df[[
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]].values
        max_length = 32

    # add BERT-required formatting; tokenize with desired BertTokenizer
    # Load Tokenizer
    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    input_ids = []
    for sentence in sentences:
        sentence_ids = tokenizer.encode(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True
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
                            sampler=sampler,
                            batch_size=batch_size
                            )
    return dataloader

# # Here is some code for testing
# test_iter = create_iters(path='./data/sem_eval_2018/test.txt',
#                          order='sequential',
#                          batch_size=64)
# for batch in test_iter:
#     print(len(batch[0]))
#     sys.exit()


def get_model():
    """
    this function loads a pre-trained bert-base-uncased model instance with a
    multi-label classification layer on top. Running it for the first time will
    take about half a minute or so, as the parameters need to be downloaded to
    your machine.
    :return:
    """
    print('Loading pre-trained BERT')
    model = BertModel.from_pretrained("bert-base-uncased")
    return model


def save_model(model, path):
    # FIXME: make model size smaller by only saving the trainable parameters
    torch.save(model, path)


def load_model(path, device):
    return torch.load(path, map_location=device)


def get_pytorch_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    return device


def parse_nonlinearity(nonlinearity_str):
    assert nonlinearity_str in ['ReLU', 'Tanh'], "Unknown nonlinearity! Please choose one of ['ReLU', 'Tanh']"
    if nonlinearity_str == 'ReLU':
        return torch.nn.ReLU()
    else:
        return torch.nn.Tanh()


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str,
                        default=os.path.join(os.getcwd(), '.data'))
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--freeze_bert', default=False, action='store_true')
    parser.add_argument('--freeze_num', type=int, default=199)
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=.1)
    args = parser.parse_args()
    return args

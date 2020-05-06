import os
from argparse import ArgumentParser
import pandas as pd
import torch
from copy import deepcopy

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
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'  # returns results already as pytorch tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Stack the input_ids, labels and attention_masks
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(labels)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Load tensors into torch Dataset object
    dataset = TensorDataset(input_ids, labels, attention_masks)
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


def bert_tokenizer(sentences, max_length=32):
    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'  # returns results already as pytorch tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Stack the input_ids, labels and attention_masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def make_dataloader(input_ids, labels, attention_masks, batch_size=16, shuffle=True):
    """ expects input_ids, labels, attention_masks to be tensors"""

    # Load tensors into torch Dataset object
    dataset = TensorDataset(input_ids, labels, attention_masks)
    # Determine what sampling mode should be used
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # Create DataLoader object
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size
                            )
    return dataloader

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


def save_model(model, unfreeze_num, snapshot_path):
    # FIXME: make model size smaller by only saving the trainable parameters
    # FIXME #2: also save optimizer state_dict, epochs, loss, etc
    # Copy instance of model to avoid mutation while training
    model_copy = deepcopy(model)

    # Delete frozen layers from model_copy instance, save state_dicts
    model_copy.encoder.encoder.layer = model_copy.encoder.encoder.layer[-(unfreeze_num):]
    print('saving BERT instance')
    torch.save({
        'BERT_state_dict': model_copy.encoder.state_dict(),
        'EMO_state_dict' : model_copy.emo_classifier.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
    }, snapshot_path)


def load_model(path, model, unfreeze_num, device):
    # Load dictionary with BERT and MLP state_dicts
    checkpoint = torch.load(path, map_location=device)
    # Overwrite last n BERT blocks, overwrite MLP params
    untuned_blocks = model.encoder.encoder.layer[-(unfreeze_num):]
    untuned_blocks = checkpoint['BERT_state_dict']
    model.emo_classifier.load_state_dict(checkpoint['EMO_state_dict'])
    return model


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
    parser.add_argument('--unfreeze_num', type=int, default=2)
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=.1)
    args = parser.parse_args()
    return args

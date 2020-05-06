import os
from argparse import ArgumentParser
import pandas as pd
import torch
import numpy as np

from copy import deepcopy

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, \
    BertModel


import sys


def create_iters(path, order, batch_size):
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
    if 'semeval' in path:
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        data_df = pd.read_csv(path, sep='\t')
        sentences = data_df.Tweet.values
        labels = data_df[[
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]].values
    elif 'sarcasm' in path:
        data_df = pd.read_json(path, lines=True)
        data_df['context'] = [l[:2] for l in data_df['context']]
        data_df['contextstr'] = ['; '.join(map(str, l)) for l in data_df['context']]
        data_df['sentence'] = data_df['response'] + data_df['contextstr']
        msk = np.random.rand(len(data_df)) < 0.8
        train = data_df[msk]
        test = data_df[~msk]
        test.to_json('./data/twitter/sarcasm_twitter_testing.json', orient='records', lines=True)
        sentences = train.sentence.values
        labels = np.where(train.label.values == 'SARCASM', 1, 0)


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
            max_length=64,
            pad_to_max_length=True
        )
        input_ids.append(torch.tensor(sentence_ids))

    # Convert input_ids and labels to tensors;
    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.tensor(labels).unsqueeze(1)

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
    print(len(model_copy._modules))
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
    parser.add_argument('--max_epochs', type=int, default=5 )
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--num_classes', type=int, default=11)
    args = parser.parse_args()
    return args

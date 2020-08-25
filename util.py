import os
from argparse import ArgumentParser
import torch
import numpy as np

from copy import deepcopy

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset

from transformers import BertModel


def bert_tokenizer(sentences, tokenizer, max_length=64):
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # returns results already as pytorch tensors
            truncation=True  # must be explicitly true since version 3.0.1 if max_length is used
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Stack the input_ids, labels and attention_masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def get_label_indicies(labels):
    """Expects labels as tensors and converts them to numpy itself"""
    labels = labels.squeeze().numpy()
    index_dict = {}
    label_distribution = {}
    unique_labels = np.unique(labels)
    for i in unique_labels:
        bin_labels = np.nonzero(labels == i)[0]
        index_dict[i] = bin_labels
        label_distribution[i] = bin_labels.size
    return index_dict, label_distribution


def split_dataset_to_support_and_query_sets(sentences, labels, masks):
    # get indices per label
    index_dict, label_distribution = get_label_indicies(labels)
    supp_indices = []
    query_indices = []
    for label in index_dict.keys():
        # if uneven number drop one example
        if label_distribution[label] % 2 != 0:
            index_dict[label] = index_dict[label][:-1]
        split_index = int(len(index_dict[label])*0.5)
        supp_indices.append(index_dict[label][:split_index])
        query_indices.append(index_dict[label][split_index:])

    # join and shuffle per label indices
    supp_indices = np.hstack(supp_indices)
    query_indices = np.hstack(query_indices)
    shuffle_idx = np.random.choice(len(supp_indices), len(supp_indices), replace=False)
    supp_indices = supp_indices[shuffle_idx]
    query_indices = query_indices[shuffle_idx]

    return \
        (sentences[supp_indices], labels[supp_indices], masks[supp_indices]), \
        (sentences[query_indices], labels[query_indices], masks[query_indices])


class EpisodicSampler(torch.utils.data.Sampler):
    """Expects TensorDataset with labels as second argument"""
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        index_dict, label_distribution = get_label_indicies(self.data_source.tensors[1])
        label_sizes = label_distribution.values()
        max_label = max(label_sizes)

        balanced_labels = []
        # expand labels with less examples than the most common one  (by duplicating x times)
        for label in index_dict.keys():
            indices = index_dict[label]
            expand = int(np.floor(max_label/label_distribution[label]))
            if expand > 1:
                expanded_indices = np.tile(indices, expand)
                # shuffle the separate tiles to avoid same pattern over iterations
                # while approximately preserving frequencies of examples
                tile_size = len(indices)
                shuffle_indices = []
                for tile in range(expand):
                    start = tile * tile_size
                    index_choice = np.arange(start, start + tile_size)
                    shuffle_indices.append(np.random.choice(index_choice, tile_size, replace=False))
                expanded_indices = expanded_indices[np.hstack(shuffle_indices)]
            else:
                expanded_indices = indices
                np.random.shuffle(expanded_indices)
            # pad labels with less examples to exactly match the number of the most common label
            remainder = max_label % label_distribution[label]
            expanded_indices = np.hstack((expanded_indices, np.random.choice(indices, remainder, replace=False)))
            balanced_labels.append(expanded_indices)

        # ensure alternating class labels
        alternating_labels = np.asarray(list(zip(*balanced_labels))).flatten()

        return iter(alternating_labels)

    def __len__(self):
        _, label_distribution = get_label_indicies(self.data_source.tensors[1])
        label_sizes = label_distribution.values()
        num_labels = len(label_sizes)
        max_label = max(label_sizes)
        return num_labels * max_label


def make_dataloader(dataset_id, input_ids, labels, attention_masks, batch_size=16, shuffle=True, episodic=True, supp_query_split=False):
    """ expects dataset_id, input_ids, labels, attention_masks to be tensors"""

    # split data into a support and query set with same label distribution and same labels at the same index
    if supp_query_split:
        supp, query = split_dataset_to_support_and_query_sets(input_ids, labels, attention_masks)
        dataset = TensorDataset(supp[0], supp[1], supp[2], query[0], query[1], query[2])
    else:
        # Load tensors into torch Dataset object
        dataset = TensorDataset(input_ids, labels, attention_masks)

    # We identify the dataset so its users can distinguish between data distributions.
    # For instance, to diminish frequency sampling between tasks that reuse the same dataset.
    dataset.id = dataset_id
    # also record number of classes
    dataset.num_classes = len(labels.unique())

    # Determine what sampling mode should be used
    if shuffle:
        if episodic:
            sampler = EpisodicSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # Create DataLoader object
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            drop_last=True
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
        'EMO_state_dict': model_copy.emo_classifier.state_dict(),
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

# FIXME: tasks are imported at this point to avoid a circular dependency:
# tasks [*] -> models [SLClassifier] -> util [parse_nonlinearity]
from tasks import *

def get_task_by_name(args, task_name):
    if 'Offenseval' == task_name:
        return OffensevalTask(cls_dim=args.mlp_dims[-1])
    elif 'SarcasmDetection' == task_name:
        return SarcasmDetection(cls_dim=args.mlp_dims[-1])
    elif 'SentimentAnalysis' == task_name:
        return SentimentAnalysis(cls_dim=args.mlp_dims[-1])
    elif 'IronySubtaskA' == task_name:
        return IronySubtaskA(cls_dim=args.mlp_dims[-1])
    elif 'IronySubtaskB' == task_name:
        return IronySubtaskB(cls_dim=args.mlp_dims[-1])
    elif 'Abuse' == task_name:
        return Abuse(cls_dim=args.mlp_dims[-1])
    elif 'Politeness' == task_name:
        return Politeness(cls_dim=args.mlp_dims[-1])
    else:
        raise ValueError('Unknown task: {}'.format(task_name))

def get_training_tasks(args):
    tasks = []
    for task_name in args.training_tasks:
        if 'SemEval18' == task_name:
            for emotion in SemEval18SingleEmotionTask.EMOTIONS:
                tasks.append(SemEval18SingleEmotionTask(emotion, cls_dim=args.mlp_dims[-1]))
        else:
            tasks.append(get_task_by_name(args, task_name))
    return tasks

def get_validation_task(args):
    return get_task_by_name(args, args.validation_task)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unfreeze_num', type=int, default=2)
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[768])
    parser.add_argument('--mlp_dropout', type=float, default=0)
    parser.add_argument('--mlp_activation', default='ReLU', choices=['ReLU', 'Tanh'])
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--bert_lr', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_test_batches', type=int, default=10)
    parser.add_argument('--episodes', type=str, default='data/sentiment_episodes_k8.pkl')
    parser.add_argument('--distance', choices=['euclidean', 'cosine'], default='euclidean')
    args = parser.parse_args()
    return args

TASK_NAMES = [
    'Abuse', 'IronySubtaskA', 'IronySubtaskB', 'Offenseval', 'Politeness',
    'SarcasmDetection', 'SemEval18', 'SentimentAnalysis',
]
def get_args_meta(args=None):
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vary_k', type=bool, default=False)
    parser.add_argument('--unfreeze_num', type=int, default=2)
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[768])
    parser.add_argument('--mlp_dropout', type=float, default=0)
    parser.add_argument('--mlp_activation', default='ReLU', choices=['ReLU', 'Tanh'])
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--meta_batch_size', type=int, default=5)
    parser.add_argument('--inner_updates', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=300)
    parser.add_argument('--bert_lr', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--inner_lr', type=float, default=1e-3)
    parser.add_argument('--custom_task_ratio', default=None)
    parser.add_argument('--num_test_batches', type=int, default=10)
    parser.add_argument('--episodes', type=str, default='data/sentiment_episodes_k8.pkl')
    parser.add_argument('--distance', choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--training_tasks', nargs='*', choices=TASK_NAMES,
        default=['SemEval18', 'Offenseval', 'SarcasmDetection'])
    parser.add_argument('--validation_task', default='SentimentAnalysis', choices=TASK_NAMES)
    return parser.parse_args(args=args)

def get_test_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str,
                        default=os.path.join(os.getcwd(), '.data'))
    parser.add_argument('--save_path', type=str, default='test_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freeze_bert', default=False, action='store_true')
    parser.add_argument('--unfreeze_num', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model', type=str, default='ProtoMAML')
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[768])
    parser.add_argument('--mlp_dropout', type=float, default=0)
    parser.add_argument('--mlp_activation', default='ReLU', choices=['ReLU', 'Tanh'])
    parser.add_argument('--task', type=str, default='SentimentAnalysis')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--episodes', type=str, default='')
    parser.add_argument('--generate_episodes', type=int, default=10)
    parser.add_argument('--num_updates', type=int, default=5)
    parser.add_argument('--num_test_batches', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_linear_with_centroids', default=False, action='store_true')
    parser.add_argument('--bert_lr', type=float, default=5e-5)
    parser.add_argument('--distance', choices=['euclidean', 'cosine'], default='euclidean')
    args = parser.parse_args()
    return args

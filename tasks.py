import math
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import sys

from models import SLClassifier
from util import bert_tokenizer, make_dataloader


def _train_dev_test_split(df):
    """
    Returns test splits 70/15/15 for the dataframe given
    """
    df_train, df_tmp = train_test_split(df, test_size=0.3, random_state=1)
    df_dev, df_test = train_test_split(df_tmp, test_size=0.5, random_state=1)
    return df_train, df_dev, df_test


class Task(object):
    r"""Base class for every task."""
    NAME = 'TASK_NAME'

    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = None

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        raise NotImplementedError

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)

    def get_name(self):
        return self.NAME

    def get_num_classes(self):
        return self.num_classes

    def describe(self):
        print('No description provided for task {}'.format(self.get_name()))

    def _get_dataframe(self, split):
        assert split in ['train', 'dev', 'test']
        if split == 'train':
            df = self.df_train
        elif split == 'dev':
            df = self.df_dev
        else:
            df = self.df_test
        return df


class TaskSamplerIter(object):
    """Iterator class used by TaskSampler."""
    def __init__(self, task_iters, method, custom_task_ratio=None):
        self.original_dataloaders = task_iters
        self.task_iters = [iter(ti) for ti in task_iters]
        self.method = method
        if custom_task_ratio is None:
            # Using the square root of the dataset size is a strategy that yields good results.
            # Additionally, we divide by the number of times the same dataset is used in
            # different tasks. This aims to attenuate bias towards the data distribution of
            # a particular dataset.
            dataset_ids = [task_iter.dataset.id for task_iter in task_iters]
            task_ratio = [math.sqrt(task_iter.dataset.tensors[0].shape[0])/dataset_ids.count(task_iter.dataset.id) for task_iter in task_iters]
        else:
            if custom_task_ratio == 'equal':
                task_num = len(self.original_dataloaders)
                task_ratio = [1/task_num] * task_num
            else:
                task_ratio = custom_task_ratio
        self.task_probs = [tr/sum(task_ratio) for tr in task_ratio]
        self.num_total_batches = sum([len(task_iter) for task_iter in task_iters])
        self.task_index = 0
        self.batch_idx = 0

    def get_task_index(self):
        return self.task_index

    def sample_next_task(self):
        if self.method == 'sequential':
            return (self.task_index + 1) % len(self.task_iters) if self.batch_idx != 0 else 0
        else:
            return np.random.choice(len(self.task_iters), p=self.task_probs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.task_iters:
            task_index = self.sample_next_task()
            task_iter = self.task_iters[task_index]

            try:
                batch = next(task_iter)
            except StopIteration:
                # Note that depending on how next it's implemented it could also
                # return an empty list instead of raising StopIteration

                # if iterator is empty initialize new iterator from original dataloader
                task_iter = iter(self.original_dataloaders[task_index])
                self.task_iters[task_index] = task_iter
                batch = next(task_iter)

            self.task_index = task_index
            self.batch_idx += 1
            if self.batch_idx == self.num_total_batches+1:
                logging.warning(
                    (
                        'Number of batches exceeds the expected amount. ' +
                        'Expected: {}; current batch idx: {}'
                    ).format(self.num_total_batches, self.batch_idx))
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return self.num_total_batches


class TaskSampler(Task):
    r"""This sampler is implemented as a task.

        task_all = TaskSampler([
                            Task_A(),
                            Task_B(),
                            Task_C(),
                        ])
        train_iter = task_all.get_iter('train')
        for batch in train_iter:
            ...
    """
    # Improvements on task sampler:
    #   - [X] Allow to specify sampling factors per task. For instance: [1, 2, 0.5, 0.5]
    #     will sample task 1 (25%), task 2 (50%) and task 3 and 4 (12.5%) each.
    #   - [X] Mind imbalance data (-> sample freq. sqrt of dataset length)
    def __init__(self, tasks, method='sequential', custom_task_ratio=None, supp_query_split=False):
        assert len(tasks) > 0
        self.tasks = tasks
        self.method = method
        self.custom_task_ratio = custom_task_ratio
        self.supp_query_split = supp_query_split

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64):
        task_iters = [task.get_iter(split, tokenizer, batch_size*task.num_classes, shuffle, random_state,
                                    supp_query_split=self.supp_query_split) for task in self.tasks]
        self._task_sampler_iter = TaskSamplerIter(task_iters, self.method, self.custom_task_ratio)

        return self._task_sampler_iter

    def _get_current_tasks(self):
        task_index = self._task_sampler_iter.get_task_index()
        return self.tasks[task_index]

    def get_task(self, task_index):
        return self.tasks[task_index]

    def get_classifier(self):
        return self._get_current_tasks.get_classifier()

    def get_loss(self, predictions, labels):
        return self._get_current_tasks().get_loss(predictions, labels)

    def calculate_accuracy(self, predictions, labels):
        return self._get_current_tasks().calculate_accuracy(predictions, labels)

    def get_name(self):
        return self._get_current_tasks().get_name()

    def get_num_classes(self):
        return self._get_current_tasks().num_classes


class SemEval18Task(Task):
    NAME = 'SemEval18'
    """
    Multi-labeled tweet data classified in 11 emotions: anger, anticipation,
    disgust, fear, joy, love, optimism, pessimism, sadness, surprise and trust.
    """
    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.emotions = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]
        self.fn_tokenizer = fn_tokenizer
        self.num_classes = len(self.emotions)
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = BCEWithLogitsLoss()

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        assert split in ['train', 'dev', 'test']
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        data_df = pd.read_csv('./data/semeval18_task1_class/{}.txt'.format(split), sep='\t')
        sentences = data_df.Tweet.values
        labels = data_df[self.emotions].values

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.type_as(predictions))

    def calculate_accuracy(self, predictions, labels):
        gold_labels = labels
        threshold = 0.5
        pred_labels = (predictions.clone().detach() > threshold).type_as(gold_labels)
        accuracy = jaccard_score(pred_labels, gold_labels, average='samples')
        return accuracy


class SemEval18SingleEmotionTask(SemEval18Task):
    """
    Serves as a single emotion tasks. It leverages the SemEval18 dataset which
    contains 11 emotions (anger, anticipation, disgust, fear, joy, love,
    optimism, pessimism, sadness, surprise and trust) creating an individual
    dataset for the single emotion task. This subset that we call single emotion
    tasks uses all the positive entries for the target emotion plus a random
    sampling of the remaining entries, creating a balanced dataset for this
    single emotion.
    """
    EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

    def __init__(self, emotion, fn_tokenizer=bert_tokenizer, cls_dim=768):
        assert emotion in self.EMOTIONS
        self.emotion = emotion
        self.emotions = [self.emotion]
        self.fn_tokenizer = fn_tokenizer
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=2)
        self.criterion = CrossEntropyLoss()
        self.num_classes = 2


    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.reshape(-1))

    def calculate_accuracy(self, predictions, labels):
        # TODO: investigate why labels is sometimes of shape [batch_size, 1] and others just [batch_size]
        # print(predictions.shape, labels.shape)
        gold_labels = torch.flatten(labels)
        n_correct = (torch.max(predictions, 1)[1].view(gold_labels.size()) == gold_labels).sum().item()
        n_total = len(gold_labels)
        return n_correct/n_total

    def get_name(self):
        return 'SemEval18_{}'.format(self.emotion)


class OffensevalTask(Task):
    NAME = 'Offenseval'
    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):

        self.fn_tokenizer = fn_tokenizer
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=2)
        self.criterion = CrossEntropyLoss()
        self.num_classes = 2

    # TODO: allow for
    # train_iter = task.get_iter('train')
    # len(train_iter) -> returns the number of batches
    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        if split == 'test' or split == 'dev':
            data_df = pd.read_csv('data/OLIDv1.0/testset-levela.tsv', sep='\t')
            sentences = data_df.tweet.values
            data_df_labels = pd.read_csv('data/OLIDv1.0/labels-levela.csv', sep=',', header=None)
            data_df_labels[1].replace(to_replace='OFF', value=1, inplace=True)
            data_df_labels[1].replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df_labels[1].values
        # TODO Make Dev set
        else:
            data_df = pd.read_csv('data/OLIDv1.0/olid-training-v1.0.tsv', sep='\t')
            sentences = data_df.tweet.values
            data_df.subtask_a.replace(to_replace='OFF', value=1, inplace=True)
            data_df.subtask_a.replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df.subtask_a.values

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
      return self.criterion(predictions, labels)

    def calculate_accuracy(self, predictions, labels):
        bin_labels = predictions.argmax(dim=1, keepdim=False) == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)


class SarcasmDetection(Task):
    NAME = 'SarcasmDetection'

    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = 2
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer


    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        assert split in ['train', 'dev', 'test']
        df = pd.read_json('data/atcs_sarcasm_data/sarcasm_twitter_{}.json'.format(split), lines=True, encoding='utf8')
        df = df.sample(frac=1).reset_index(drop=True)

        df['context'] = [l[:2] for l in df['context']]
        df['contextstr'] = ['; '.join(map(str, l)) for l in df['context']]
        df['sentence'] = df['response'] + df['contextstr']

        sentences = df.sentence.values
        labels = np.where(df.label.values == 'SARCASM', 1, 0)

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)#.unsqueeze(1)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)



class SentimentAnalysis(Task):
    NAME = 'SentimentAnalysis'

    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = 2
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer
        df = pd.read_csv('data/sem_eval_2015/tweets_output.txt',header=None, sep='\t', names=['ID1', 'ID2', 'label','sentence'])
        df = df[df.label != 'neutral']
        df = df[df.label != 'objective']
        self.df = df[df.label != 'objective-OR-neutral']
        self.df_train, self.df_dev, self.df_test = _train_dev_test_split(self.df)

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        # current iter will have only two classes; we could extend it to have more
        df = self._get_dataframe(split)

        sentences = df.sentence.values
        labels = np.where(df.label.values == 'positive', 1, 0)

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)#.unsqueeze(1)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)


class IronySubtaskA(Task):
    NAME = 'IronySubtaskA'

    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = 2
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer
        self.df = pd.read_csv('data/sem_eval_2018/SemEval2018-T3-train-taskA.txt', sep='\t', header=0, names=['Tweet_index', 'Label', 'Tweet_text'])
        self.df_train, self.df_dev, self.df_test = _train_dev_test_split(self.df)

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        # current iter will have only two classes; we could extend it to have more
        df = self._get_dataframe(split)

        sentences = df.Tweet_text.values
        labels = np.where(df.Label.values == 1, 1, 0)

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)#.unsqueeze(1)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)

#TODO: right now this task has 4 categories; we could possibly remove one of four categories if the task is too difficult
class IronySubtaskB(Task):
    NAME = 'IronySubtaskB'

    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = 4
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer
        self.df = pd.read_csv('data/sem_eval_2018/SemEval2018-T3-train-taskB.txt', sep='\t', header=0, names=['Tweet_index', 'Label', 'Tweet_text'])
        self.df_train, self.df_dev, self.df_test = _train_dev_test_split(self.df)

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        # current iter will have only two classes; we could extend it to have more
        df = self._get_dataframe(split)

        sentences = df.Tweet_text.values
        labels = df.Label.values

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)#.unsqueeze(1)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)

class Abuse(Task):
    NAME = 'Abuse'

    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = 3
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer
        self.df = pd.read_csv('data/tweet_wassem/twitter_data_waseem_hovy.csv', sep=',', header=0, names=['Tweet_index', 'Tweet_text', 'Label'])
        self.df_train, self.df_dev, self.df_test = _train_dev_test_split(self.df)

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        # current iter will have only two classes; we could extend it to have more
        df = self._get_dataframe(split)

        sentences = df.Tweet_text.values
        labels = df.Label.values

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)#.unsqueeze(1)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)


class Politeness(Task):
    NAME = 'Politeness'
    """
    Stanford Politeness Corpus (Wikipedia). Original annotations: 1 = Polite; 0 = Neutral; -1 = impolite.
    Classes: Impolite(0), Neutral(1), Polite(2)
    """
    def __init__(self, fn_tokenizer=bert_tokenizer, cls_dim=768):
        self.num_classes = 3
        self.classes = {
            0: 'Impolite',
            1: 'Neutral',
            2: 'Polite'
        }
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer
        self.df = pd.read_csv('data/stanford_politeness_2013/wikipedia-politeness-corpus.csv')
        self.df['annotation'] = self.df.label  # Original classification {-1, 0, 1}
        # Due to the use of the CrossEntropyLoss we need the labels to represent indexes (>=0).
        # Hence we move our labels one up from {-1, 0, 1} to {0, 1, 2}.
        self.df.label = self.df.label + 1
        self.df_train, self.df_dev, self.df_test = _train_dev_test_split(self.df)

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64, supp_query_split=False):
        assert split in ['train', 'dev', 'test']
        if split == 'train':
            df = self.df_train
        elif split == 'dev':
            df = self.df_dev
        else:
            df = self.df_test
        input_ids, attention_masks = self.fn_tokenizer(df.text, tokenizer, max_length=max_length)
        labels = torch.tensor(df.label.values)
        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle, supp_query_split=supp_query_split)

    def describe(self):
        df = self.df
        print('Task {}, split(70/15/15)'.format(self.get_name()))
        print('\tClasses: {}'.format(self.classes))
        print('\tExamples:')
        dist = df.label.value_counts().to_dict()
        for label in sorted(dist.keys()):
            print('\t\t{}: {} ({:.2%})'.format(self.classes[label], dist[label], dist[label]/len(df)))
        text_desc = df.text.apply(lambda x: len(x.split(' '))).describe()
        print('\tText lengths: {:.2f} +/- {:.2f}; [{} (min), {} (25%), {} (50%), {} (75%), {} (max)]'.format(
            text_desc['mean'], text_desc['std'], text_desc['min'],
            text_desc['25%'], text_desc['50%'], text_desc['75%'], text_desc['max']))

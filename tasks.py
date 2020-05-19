import math
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
import numpy as np
import logging
import sys

from models import SLClassifier
from util import bert_tokenizer, make_dataloader


class Task(object):
    r"""Base class for every task."""
    NAME = 'TASK_NAME'

    def __init__(self):
        self.num_classes = None

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1):
        raise NotImplementedError

    def get_classifier(self):
        raise NotImplementedError

    def get_loss(self, predictions, labels):
        raise NotImplementedError

    def calculate_accuracy(self, predictions, labels):
        raise NotImplementedError

    def get_name(self):
        return self.NAME

    def get_num_classes(self):
        return self.num_classes


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

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=32):
        task_iters = [task.get_iter(split, tokenizer, batch_size, shuffle, random_state,
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
    def __init__(self, fn_tokenizer=bert_tokenizer):
        self.emotions = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]
        self.fn_tokenizer = fn_tokenizer
        self.num_classes = len(self.emotions)
        self.classifier = SLClassifier(target_dim=self.num_classes)
        self.criterion = BCEWithLogitsLoss()

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=32, supp_query_split=False):
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

    def __init__(self, emotion, fn_tokenizer=bert_tokenizer):
        assert emotion in self.EMOTIONS
        self.emotion = emotion
        self.emotions = [self.emotion]
        self.fn_tokenizer = fn_tokenizer
        self.classifier = SLClassifier(target_dim=2)
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
    def __init__(self, fn_tokenizer=bert_tokenizer):

        self.fn_tokenizer = fn_tokenizer
        self.classifier = SLClassifier(target_dim=2)
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

    def __init__(self, fn_tokenizer=bert_tokenizer):
        self.num_classes = 2
        self.classifier = SLClassifier(target_dim=self.num_classes)
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

    def __init__(self, fn_tokenizer=bert_tokenizer):
        self.num_classes = 2
        self.classifier = SLClassifier(target_dim=self.num_classes)
        self.criterion = CrossEntropyLoss()
        self.fn_tokenizer = fn_tokenizer

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        # current iter will have only two classes; we could extend it to have more
        df = pd.read_csv('data/sem_eval_2015/tweets_output.txt',header=None, sep='\t', names=['ID1', 'ID2', 'label','sentence'])
        df = df[df.label != 'neutral']
        df = df[df.label != 'objective']
        df = df[df.label != 'objective-OR-neutral']

        sentences = df.sentence.values
        labels = np.where(df.label.values == 'positive', 1, 0)

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)#.unsqueeze(1)

        return make_dataloader(self.NAME, input_ids, labels, attention_masks, batch_size, shuffle)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.long())

    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)

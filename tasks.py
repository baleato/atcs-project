import math
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
import numpy as np
import logging

from models import MLPClassifier
from util import bert_tokenizer, make_dataloader


class Task(object):
    r"""Base class for every task."""
    NAME = 'TASK_NAME'

    def __init__(self):
        pass

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


class TaskSamplerIter(object):
    """Iterator class used by TaskSampler."""
    def __init__(self, task_iters, method, custom_task_ratio=None):
        self.original_dataloaders = task_iters
        self.task_iters = [iter(ti) for ti in task_iters]
        self.method = method
        if custom_task_ratio is None:
            task_ratio = [math.sqrt(len(task_iter)) for task_iter in task_iters]
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
            if self.batch_idx > self.num_total_batches:
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

class MixedTaskSamplerIter(TaskSamplerIter):
    """Iterator class used by TaskSampler.
    Batch consists of multiple tasks.
    Returns batch (list) of length 4 with
    batch[0]: token ids, batch[1]: labels, batch[2]: attention_masks, batch[4]: task ids
    """
    def __init__(self, task_iters, batch_size, method, custom_task_ratio=None):
        super(MixedTaskSamplerIter, self).__init__(task_iters, method, custom_task_ratio)
        self.num_total_batches = int(np.ceil(self.num_total_batches/batch_size))
        self.batch_size = batch_size

    def __next__(self):
        if self.task_iters:
            task_selection = []
            for b in range(self.batch_size):
                self.task_index = self.sample_next_task()
                task_iter = self.task_iters[self.task_index]
                try:
                    task_sample = next(task_iter)
                    # ensure consistent label size
                    task_sample[1] = task_sample[1].view(1)
                    task_sample.append(torch.tensor(self.task_index).view(1))
                except StopIteration:
                    # Note that depending on how next it's implemented it could also
                    # return an empty list instead of raising StopIteration

                    # if iterator is empty initialize new iterator from original dataloader
                    task_iter = iter(self.original_dataloaders[self.task_index])
                    task_sample = next(task_iter)
                    task_sample[1] = task_sample[1].view(1)
                    task_sample.append(torch.tensor(self.task_index).view(1))
                task_selection.append(task_sample)
            batch = []
            for i in range(len(task_sample)):
                bla = [row[i] for row in task_selection]
                field = torch.cat([row[i].long() for row in task_selection])
                batch.append(field)
            self.batch_idx += 1
            if self.batch_idx > self.num_total_batches:
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
    #   - [X] Mix different task examples within a batch
    #   - [X] Allow to specify sampling factors per task. For instance: [1, 2, 0.5, 0.5]
    #     will sample task 1 (25%), task 2 (50%) and task 3 and 4 (12.5%) each.
    #   - [X] Mind imbalance data (-> sample freq. sqrt of dataset length)
    def __init__(self, tasks, method='sequential', custom_task_ratio=None, mixed_batch=False):
        assert len(tasks) > 0
        self.tasks = tasks
        self.method = method
        self.custom_task_ratio = custom_task_ratio
        self.mixed_batch = mixed_batch

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=32):
        if self.mixed_batch:
            task_iters = [task.get_iter(split, tokenizer, 1, shuffle, random_state, max_length) for task in self.tasks]
            self._task_sampler_iter = MixedTaskSamplerIter(task_iters, batch_size, self.method, self.custom_task_ratio)
        else:
            task_iters = [task.get_iter(split, tokenizer, batch_size, shuffle, random_state) for task in self.tasks]
            self._task_sampler_iter = TaskSamplerIter(task_iters, self.method, self.custom_task_ratio)
        return self._task_sampler_iter

    def _get_current_tasks(self):
        task_index = self._task_sampler_iter.get_task_index()
        return self.tasks[task_index]

    def get_classifier(self):
        return self._get_current_tasks.get_classifier()

    def get_loss(self, predictions, labels):
        return self._get_current_tasks().get_loss(predictions, labels)

    def calculate_accuracy(self, predictions, labels):
        return self._get_current_tasks().calculate_accuracy(predictions, labels)

    def get_name(self):
        return self._get_current_tasks().get_name()


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
        self.classifier = MLPClassifier(target_dim=len(self.emotions))
        self.criterion = BCEWithLogitsLoss()

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=32):
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

        return make_dataloader(input_ids, labels, attention_masks, batch_size, shuffle)

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
    def __init__(self, emotion, fn_tokenizer=bert_tokenizer):
        self.emotion = emotion
        self.emotions = [self.emotion]
        self.fn_tokenizer = fn_tokenizer
        self.classifier = MLPClassifier(target_dim=2)
        self.criterion = CrossEntropyLoss()

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=32):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        assert split in ['train', 'dev', 'test']
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        df = pd.read_csv('./data/semeval18_task1_class/{}.txt'.format(split), sep='\t')
        # select all positive labels for the emotion and an equally sized sample of negative sentences
        df_emotion = df[df[self.emotion] == 1].copy()
        df_other = df[df[self.emotion] == 0].sample(df_emotion.shape[0], random_state=1)
        selected_df = pd.concat([df_emotion, df_other]).sample(frac=1, random_state=1)
        sentences = selected_df.Tweet.values
        labels = selected_df[self.emotions].values

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)

        return make_dataloader(input_ids, labels, attention_masks, batch_size, shuffle)

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.reshape(-1))

    def calculate_accuracy(self, predictions, labels):
        # TODO: investigate why labels is sometimes of shape [batch_size, 1] and others just [batch_size]
        # print(predictions.shape, labels.shape)
        gold_labels = torch.flatten(labels)
        n_correct = (torch.max(predictions, 1)[1].view(gold_labels.size()) == gold_labels).sum().item()
        n_total = len(gold_labels)
        return 100. * n_correct/n_total

    def get_name(self):
        return 'SemEval18_{}'.format(self.emotion)


class SemEval18AngerTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18Anger'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18AngerTask, self).__init__('anger', fn_tokenizer)

class SemEval18AnticipationTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18Anticipation'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18AnticipationTask, self).__init__('anticipation', fn_tokenizer)

class SemEval18SurpriseTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18Surprise'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18SurpriseTask, self).__init__('surprise', fn_tokenizer)

class SemEval18TrustTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18Trust'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18TrustTask, self).__init__('trust', fn_tokenizer)


class OffensevalTask(Task):
    NAME = 'Offenseval'
    def __init__(self, fn_tokenizer=bert_tokenizer):

        self.fn_tokenizer = fn_tokenizer
        self.classifier = MLPClassifier(target_dim=2)
        self.criterion = CrossEntropyLoss()

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64):
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        if split == 'test':
            data_df = pd.read_csv('data/offenseval/testset-levela.csv', sep='\t')
            sentences = data_df.tweet.values
            data_df_labels = pd.read_csv('data/offenseval/labels-levela.csv', sep=',', header=None)
            data_df_labels[1].replace(to_replace='OFF', value=1, inplace=True)
            data_df_labels[1].replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df_labels[1].values
        # TODO Make Dev set
        else:
            data_df = pd.read_csv('data/offenseval/offenseval-training-v1.tsv', sep='\t')
            sentences = data_df.tweet.values
            data_df.subtask_a.replace(to_replace='OFF', value=1, inplace=True)
            data_df.subtask_a.replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df.subtask_a.values

        input_ids, attention_masks = self.fn_tokenizer(sentences, tokenizer, max_length=max_length)
        labels = torch.tensor(labels)

        return make_dataloader(input_ids, labels, attention_masks, batch_size, shuffle)

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
        self.classifier = MLPClassifier(target_dim=1)
        self.criterion = BCEWithLogitsLoss()
        self.fn_tokenizer = fn_tokenizer

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64):
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
        labels = torch.tensor(labels).unsqueeze(1)

        return make_dataloader(input_ids, labels, attention_masks, batch_size, shuffle)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.type_as(predictions).reshape_as(predictions))

    def calculate_accuracy(self, predictions, labels):
        pred_labels = torch.sigmoid(predictions).round()
        bin_labels = pred_labels == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)


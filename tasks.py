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
    NAME = 'TASK_NAME'
    def __init__(self):
        pass

    # TODO: allow for
    # train_iter = task.get_iter('train')
    # len(train_iter) -> returns the number of batches
    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1):
        raise NotImplementedError

    def get_num_batches(self, split, batch_size=1):
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
    def __init__(self, task_iters):
        self.task_iters = [iter(ti) for ti in task_iters]
        self._len_tasks_called = sum([len(task_iter) for task_iter in task_iters ])
        self.task_indexes = list(range(len(task_iters)))
        task_num_examples = [len(task_iter) for task_iter in task_iters]
        total_num_examples = sum(task_num_examples)
        self.task_index = 0
        batch = []
        self.task_indexes_index = 0
        self.batch_idx = 0

    def get_task_index(self):
        return self.task_index

    def __iter__(self):
        return self

    def __next__(self):
        while self.task_iters:
            task_iter = self.task_iters[self.task_indexes_index]
            task_index = self.task_indexes[self.task_indexes_index]
            try:
                batch = next(task_iter)
            except StopIteration:
                # Note that depending on how next it's implemented it could also
                # return an empty list instead of raising StopIteration
                batch = []
            if not batch:
                self.task_iters.remove(task_iter)
                self.task_indexes.remove(task_index)
                if self.task_indexes:
                    self.task_indexes_index = self.task_indexes_index % len(self.task_indexes)
            else:
                self.task_index = task_index
                self.task_indexes_index = (self.task_indexes_index + 1) % len(self.task_indexes)
                self.batch_idx += 1
                if self.batch_idx > self._len_tasks_called:
                    logging.warning(
                        (
                            'Number of batches exceeds the expected amount. ' + \
                            'Expected: {}; current batch idx: {}'
                        ).format(self._len_tasks_called, selfbatch_idx))
                return batch

        raise StopIteration

    def __len__(self):
        return self._len_tasks_called




# TODO: implement a better sampler:
#   - Mix different tasks within a batch
#   - Allow to specify sampling factors per task. For instance: [1, 2, 0.5, 0.5]
#     will sample task 1 (25%), task 2 (50%) and task 3 and 4 (12.5%) each.
#   - Mind imbalance data (-> sample freq. sqrt of dataset length)
class TaskSampler(Task):
    """
    Args:
        tasks: Task's list
    """
    def __init__(self, tasks):
        assert len(tasks) > 0
        self.tasks = tasks

    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=32):
        """
        Combines batches from different tasks
        """
        # TODO: pass task params along
        task_iters = [task.get_iter(split, tokenizer) for task in self.tasks]
        self._task_sampler_iter = TaskSamplerIter(task_iters)
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
        self.num_classes = 2


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
        return 'SemEval18{}'.format(self.emotion)


class SemEval18AngerTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18anger'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18AngerTask, self).__init__('anger', fn_tokenizer)

class SemEval18AnticipationTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18anticipation'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18AnticipationTask, self).__init__('anticipation', fn_tokenizer)

class SemEval18SurpriseTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18surprise'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18SurpriseTask, self).__init__('surprise', fn_tokenizer)

class SemEval18TrustTask(SemEval18SingleEmotionTask):
    NAME = 'SemEval18trust'
    def __init__(self, fn_tokenizer=bert_tokenizer):
        super(SemEval18TrustTask, self).__init__('trust', fn_tokenizer)


class OffensevalTask(Task):
    NAME = 'Offenseval'
    def __init__(self, fn_tokenizer=bert_tokenizer):

        self.fn_tokenizer = fn_tokenizer
        self.classifier = MLPClassifier(target_dim=2)
        self.criterion = CrossEntropyLoss()
        self.num_classes = 2

    # TODO: allow for
    # train_iter = task.get_iter('train')
    # len(train_iter) -> returns the number of batches
    def get_iter(self, split, tokenizer, batch_size=16, shuffle=False, random_state=1, max_length=64):
        # Load dataset into Pandas Dataframe, then extract columns as numpy arrays
        if split == 'test':
            data_df = pd.read_csv('data/offenseval/testset-levela.csv', sep='\t')
            sentences = data_df.tweet.values
            data_df_labels = pd.read_csv('data/offenseval/labels-levela.csv', sep=',', header=None)
            data_df_labels[1].replace(to_replace='OFF', value=1, inplace=True)
            data_df_labels[1].replace(to_replace='NOT', value=0, inplace=True)
            labels = data_df_labels[1].values
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

    def __init__(self, fn_tokenizer=None):
        self.splits = {}
        self.classifier = MLPClassifier(target_dim=1)
        self.criterion = BCEWithLogitsLoss()
        self.fn_tokenizer = fn_tokenizer
        self.num_classes = 2
        for split in ['train', 'dev', 'test']:
            self.splits.setdefault(
                split,
                pd.read_json('data/atcs_sarcasm_data/sarcasm_twitter_{}.json'.format(split), lines=True, encoding='utf8'))

    def get_iter(self, split, batch_size=16, shuffle=False, random_state=1):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        assert split in ['train', 'dev', 'test']
        df = self.splits.get(split)
        df = df.sample(frac=1).reset_index(drop=True)
        ix = 0
        while ix < len(df):
            df_batch = df.iloc[ix:ix+batch_size].copy(deep=True)

            df_batch['context'] = [l[:2] for l in df_batch['context']]
            df_batch['contextstr'] = ['; '.join(map(str, l)) for l in df_batch['context']]
            df_batch['sentence'] = df_batch['response'] + df_batch['contextstr']

            sentences = df_batch.sentence.values
            labels = np.where(df_batch.label.values == 'SARCASM', 1, 0)


            if self.fn_tokenizer:
                sentences = self.fn_tokenizer(list(sentences))
            yield sentences, torch.tensor(labels).unsqueeze(1)
            ix += batch_size

    def get_num_batches(self, split, batch_size=1):
        assert split in ['train', 'dev', 'test']
        return math.ceil(len(self.splits.get(split))/batch_size)

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels.type_as(predictions))

    def calculate_accuracy(self, predictions, labels):
        gold_labels = labels
        threshold = 0.5
        pred_labels = (predictions.clone().detach() > threshold).type_as(gold_labels)
        accuracy = accuracy_score(gold_labels, pred_labels)
        return accuracy

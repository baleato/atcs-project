import math
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss
from sklearn.metrics import jaccard_score, f1_score

from models import MLPClassifier
from util import bert_tokenizer, make_dataloader

# TODO
class TaskSampler(object):
    """
    Args:
        tasks: Task's list
        freq_factors: list of sampling frequency factors by task
        ...
    """
    pass


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
        gold_labels = labels[:, 0]
        n_correct = (torch.max(predictions, 1)[1].view(gold_labels.size()) == gold_labels).sum().item()
        n_total = len(gold_labels)
        return 100. * n_correct/n_total


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

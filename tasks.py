import pandas as pd

from util import create_iters

class TaskSampler(object):
    """
    Args:
        tasks: Task's list
        freq_factors: list of sampling frequency factors by task
    """
    def __init__(self, tasks, batch_size):
        pass


class Task(object):
    """

    """
    def __init__(self):
        pass

    def get_classifier(self):
        pass




# TODO:
# - Use semeval18_task1_class as 11 different tasks
# - Train on 10 and test in 1
class DataLoader(object):

    def __init__(self, batch_size, num_samples_per_class):
        """
        Args:
            batch_size
            num_samples_per_class
        """
        pass


    # TODO: think how to iterate over batches, if to use iterables
    def next_batch(self, batch_type):
        """
        Iterates over batches
        Args:
            batch_type: train/val/test
        Returns:
            batch or
        """
        raise NotImplementedError


class MyIter:
    def __init__(self):
        self.prev = 0
        self.curr = 1

    def __iter__(self):
        return self

    def __next__(self):
        value = self.curr
        self.curr += self.prev
        self.prev = value
        return value


class SemEval18Task(Task):
    """
    Multi-labeled tweet data classified in 11 emotions: anger, anticipation,
    disgust, fear, joy, love, optimism, pessimism, sadness, surprise and trust.
    """
    def __init__(self, fn_tokenizer=None):
        self.emotions = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]
        self.fn_tokenizer = fn_tokenizer
        # TODO:
        # - pre-process dataset

    def get_iter(self, split, batch_size=16, shuffle=False, random_state=1):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        df = pd.read_table('data/semeval18_task1_class/{}.txt'.format(split))
        if shuffle:
            df = df.sample(frac=1, random_state=1)

        ix = 0
        while ix < len(df):
            df_batch = df.iloc[ix:ix+batch_size]
            sentences = df_batch.Tweet.values
            labels = df_batch[self.emotions].values
            if self.fn_tokenizer:
                input_ids = []
                sentences = self.fn_tokenizer(list(sentences))
            yield sentences, labels
            ix += batch_size

class SemEval18SingleEmotionTask(Task):
    """
    Serves as a single emotion tasks. It leverages the SemEval18 dataset which
    contains 11 emotions (anger, anticipation, disgust, fear, joy, love,
    optimism, pessimism, sadness, surprise and trust) creating an individual
    dataset for the single emotion task. This subset that we call single emotion
    tasks uses all the positive entries for the target emotion plus a random
    sampling of the remaining entries, creating a balanced dataset for this
    single emotion.
    """
    def __init__(self, emotion, num_samples=16):
        self.emotion = emotion
        self.num_samples = num_samples

    def get_iter(self, split, batch_size=16):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        assert split in ['train', 'dev', 'test']

        df = pd.read_table('data/semeval18_task1_class/{}.txt'.format(split))
        df_emotion = df[df[self.emotion] == 1]
        df_other = df[df[self.emotion] == 0].sample(df_emotion.shape[0])
        df = pd.concat([df_emotion, df_other]).sample(frac=1, random_state=1)
        ix = 0
        while ix < len(df):
            df_batch = df.iloc[ix:ix+batch_size]
            sentences = df_batch.Tweet.values
            labels = df_batch[self.emotion].values
            yield sentences, labels
            ix += batch_size


class SemEval18AngerTask(SemEval18SingleEmotionTask):
    def __init__(self):
        super(SemEval18AngerTask, self).__init__('anger')

import unittest
import torch
from transformers import BertTokenizer
import torch

import tasks


class DummyModel(torch.nn.Module):
    def forward(self, inputs, attention_mask=None):
        return torch.rand(len(inputs), 768)


class TestModels(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def _test_task_iters(self, task, splits=['train', 'dev', 'test']):
        for split in splits:
            task_iter = task.get_iter(split, self.tokenizer)
            print('{} {} {}'.format(task.get_name(), split, len(task_iter)))

    def _test_task(self, task):
        classifier = task.get_classifier()
        task_iter = task.get_iter('train', self.tokenizer)
        model = DummyModel()
        for batch in task_iter:
            sentences = batch[0]
            labels = batch[1]
            attention_masks = batch[2]

            output = model(sentences, attention_masks)
            predictions = classifier(output)
            loss = task.get_loss(predictions, labels)
            acc = task.calculate_accuracy(predictions, labels)
            self.assertTrue(0 <= acc <= 1)
            break
        task.describe()

    def test_Tasks(self):
        for task in [
                tasks.SemEval18Task(),
                tasks.SemEval18SingleEmotionTask('anger'),
                tasks.OffensevalTask(),
                tasks.SarcasmDetection(),
                tasks.SentimentAnalysis(),
                tasks.IronySubtaskA(),
                tasks.IronySubtaskB(),
                tasks.Abuse(),
                tasks.Politeness()]:
            self._test_task(task)
            self._test_task_iters(task, ['train'])


    def test_Tasks2(self):
        tasks_array = [
            tasks.SemEval18SingleEmotionTask('fear', data_path='./data/semeval18_fear_train.txt'),
            tasks.SemEval18SingleEmotionTask('optimism', data_path='./data/semeval18_optimism_train.txt'),
            tasks.SemEval18SingleEmotionTask('sadness', data_path='./data/semeval18_sadness_train.txt'),
            tasks.SarcasmDetection(),
            tasks.IronySubtaskA(),
            tasks.Politeness()
        ]
        for task in tasks_array:
            self._test_task_iters(task, ['train'])
        sampler = tasks.TaskSampler(tasks_array, supp_query_split=True, avoid_repetition=True)
        train_iter = sampler.get_iter('train', tokenizer=self.tokenizer, batch_size=16, shuffle=True)
        self.assertEqual(int(1089 / 16) * len(tasks_array), len(train_iter))


class DummyTask(tasks.Task):
    def __init__(self, iterable):
        self.iterable = iterable

    def get_iter(self, *args, **argv):
        return self.iterable


class TestSamplers(unittest.TestCase):
    def test_task_sampler_episodic_fashion(self):
        taskA = DummyTask([0, 2])
        taskB = DummyTask([1, 3])
        sampler = tasks.TaskSampler([taskA, taskB], custom_task_ratio=[0.5, 0.5])
        sampler_iter = sampler.get_iter(split=None, tokenizer=None)
        batches = []
        for i in range(len(sampler_iter) * 2):
            batches.append(next(sampler_iter))
        self.assertEqual(batches, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_task_sampler_custom_ratio(self):
        taskA = DummyTask([0, 2, 4, 6])
        taskB = DummyTask([1, 3])
        sampler = tasks.TaskSampler([taskA, taskB], custom_task_ratio=[0.5, 0.5])
        sampler_iter = sampler.get_iter(split=None, tokenizer=None)
        batches = []
        for i in range(len(sampler_iter) * 2):
            batches.append(next(sampler_iter))
        self.assertEqual(batches, [0, 1, 2, 3, 4, 1, 6, 3, 0, 1, 2, 3])

    def test_task_sampler_episodic_fashion_avoiding_repetition(self):
        taskA = DummyTask([0, 2, 4, 6])
        taskB = DummyTask([1, 3])
        sampler = tasks.TaskSampler([taskA, taskB], custom_task_ratio=[0.5, 0.5], avoid_repetition=True)
        sampler_iter = sampler.get_iter(split=None, tokenizer=None)
        batches = []
        for i in range(len(sampler_iter) * 2):
            batches.append(next(sampler_iter))
        self.assertEqual(batches, [0, 1, 2, 3, 0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()

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

    def _test_task_iters(self, task):
        for split in ['train', 'dev', 'test']:
            task_iter = task.get_iter(split, self.tokenizer)
            print('{} {} {}'.format(task.get_name(), split, len(task_iter)))

    def _test_task(self, task):
        classifier = task.get_classifier()
        task_iter = task.get_iter('dev', self.tokenizer)
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
            # self._test_task(task)
            self._test_task_iters(task)


if __name__ == '__main__':
    unittest.main()

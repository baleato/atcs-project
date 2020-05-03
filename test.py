import unittest
import torch
from transformers import BertTokenizer

from tasks import SemEval18AngerTask, SemEval18Task

class TestStringMethods(unittest.TestCase):

    def test_task(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        print('Loading Tokenizer..')
        def fn_tokenizer(sentences):
            input_ids = []
            for sentence in sentences:
                sentence_ids = tokenizer.encode(
                    sentence,
                    add_special_tokens=True,
                    max_length=32,
                    pad_to_max_length=True
                )
                input_ids.append(torch.tensor(sentence_ids))

            # Convert input_ids and labels to tensors;
            return torch.stack(input_ids, dim=0)
        batch_size = 4
        task = SemEval18Task(fn_tokenizer=fn_tokenizer)
        batch_size = 16
        split_iter = task.get_iter('dev', batch_size=batch_size)
        for batch_index, (sentences, labels) in enumerate(split_iter):
            self.assertEqual(len(sentences), len(labels))
            self.assertTrue(len(sentences) <= batch_size)
        print(batch_index)
        print(sentences, sentences.shape)
        print(labels)
        self.assertTrue(batch_index > 0)

if __name__ == '__main__':
    unittest.main()

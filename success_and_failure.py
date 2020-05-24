import os
import torch
from transformers import BertTokenizer
from tasks import *

def get_success_and_failure_cases(task, tokenizer, path='predictions/'):
    mtl_preds = torch.load(open(os.path.join(path, 'predictions_{}_{}.pkl'.format('MTL', task.get_name())), "rb"))
    proto_preds = torch.load(open(os.path.join(path, 'predictions_{}_{}.pkl'.format('ProtoNet', task.get_name())), "rb"))
    maml_preds = torch.load(open(os.path.join(path, 'predictions_{}_{}.pkl'.format('ProtoMAML', task.get_name())), "rb"))

    raw_preds = torch.stack([mtl_preds['raw'], proto_preds['raw'], maml_preds['raw']], dim=2).cpu()
    class_preds = torch.stack([mtl_preds['class'], proto_preds['class'], maml_preds['class']], dim=1).cpu()

    results_dict = {}

    test_iter = task.get_iter('test', tokenizer, shuffle=False)
    sentences = test_iter.dataset.tensors[0]
    labels = test_iter.dataset.tensors[1]
    correct = class_preds == labels.unsqueeze(dim=1)

    success = correct.all(dim=1)
    results_dict['success'] = (sentences[success], labels[success], class_preds[success], raw_preds[success])

    failure = ~correct.any(dim=1)
    results_dict['failure'] = (sentences[failure], labels[failure], class_preds[failure], raw_preds[failure])

    difference = correct.any(dim=1) & ~success
    results_dict['difference'] = (sentences[difference], labels[difference], class_preds[difference], raw_preds[difference])

    return results_dict


def parse_entry(tokenizer, result_dict_field, idx=None):
    if idx is None:
        idx = range(result_dict_field[0].shape[0])
    elif isinstance(idx, list):
        idx = idx
    else:
        idx = [idx]

    for i in idx:
        print(tokenizer.decode(result_dict_field[0][i]))
        print('Label: {}, Prediction: {}, Detailed: {}'.format(
            result_dict_field[1][i],
            result_dict_field[2][i],
            result_dict_field[3][i].t()))
        print()


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    results_dict = get_success_and_failure_cases(SentimentAnalysis(), tokenizer)
    parse_entry(tokenizer, results_dict['difference'])

from collections import deque

import torch.nn as nn
from transformers import BertModel


# class MultiTaskClassifier(nn.Module):
#     def __init__(self, config):
#         super(MetaLearner, self).__init__()
#         self.encoder = BertModel.from_pretrained("bert-base-uncased")
#         self.encoder.requires_grad_(False)
#         self.classifier_map = {}
#         # TODO: unfreeze top n layers
#         # top_n_bert_layers = deque(
#         #    self.encoder.parameters(),
#         #    maxlen=config.n_layers_bert_trained)
#         # for params in top_n_bert_layers:
#         #   params.requires_grad = True
#
#     def add_task(self, task_name, classifier):
#         self.classifier_map
#
#     def forward(self, sentences):
#         encoded = self.encoder(sentences)[0]
#         return encoded[:, 0, :]

class MetaLearner(nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.requires_grad_(False)
        # TODO: unfreeze top n layers
        # top_n_bert_layers = deque(
        #    self.encoder.parameters(),
        #    maxlen=config.n_layers_bert_trained)
        # for params in top_n_bert_layers:
        #   params.requires_grad = True

    def forward(self, inputs, task_name=None):
        task_module_name = 'task_{}'.format(task_name)
        assert task_module_name in self._modules

        encoded = self.encoder(inputs)[0]
        cls_token_enc = encoded[:, 0, :]
        classifier = self._modules[task_module_name]
        return classifier(cls_token_enc)


    def add_task_classifier(self, task_name, classifier):
        assert issubclass(type(classifier), nn.Module)
        self.add_module('task_{}'.format(task_name), classifier)



class MLPClassifier(nn.Module):
    """
    Class for Multi-Layer Perceptron Classifier
    """
    def __init__(self, input_dim=768, target_dim=2):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, target_dim)
        )

    def forward(self, input):
        output = self.network(input)
        return output

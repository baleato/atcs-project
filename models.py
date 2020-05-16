from collections import deque

import torch.nn as nn
from transformers import BertModel
from util import parse_nonlinearity
import torch
from copy import deepcopy


class MetaLearner(nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.requires_grad_(False)
        for block in self.encoder.encoder.layer[-(config.unfreeze_num):]:
            for params in block.parameters():
                params.requires_grad = True

    def forward(self, inputs, task_name=None, attention_mask=None):
        task_module_name = 'task_{}'.format(task_name)
        assert task_module_name in self._modules

        encoded = self.encoder(inputs, attention_mask=attention_mask)[0]
        cls_token_enc = encoded[:, 0, :]
        classifier = self._modules[task_module_name]
        return classifier(cls_token_enc)

    def add_task_classifier(self, task_name, classifier):
        assert issubclass(type(classifier), nn.Module)
        self.add_module('task_{}'.format(task_name), classifier)

    def save_model(self, unfreeze_num, snapshot_path):
        # FIXME #2: also save optimizer state_dict, epochs, loss, etc
        # Copy instance of model to avoid mutation while training
        bert_model_copy = deepcopy(self.encoder)

        # Delete frozen layers from model_copy instance, save state_dicts
        state_dicts = {'unfreeze_num': unfreeze_num}
        for module in self._modules:
            if module == 'encoder':
                for i in range(1, unfreeze_num+1):
                    state_dicts['bert_l_-{}'.format(i)] = bert_model_copy.encoder.layer[-i].state_dict()
            if 'task' in module:
                state_dicts[module+'_state_dict'] = self._modules[module].state_dict()
        torch.save(state_dicts, snapshot_path)


    def load_model(self, path, device):
        # Load dictionary with BERT and MLP state_dicts
        checkpoint = torch.load(path, map_location=device)
        unfreeze_num = checkpoint['unfreeze_num']
        # Overwrite last n BERT blocks, overwrite MLP params
        for i in range(1, unfreeze_num + 1):
            self.encoder.encoder.layer[-i].load_state_dict(checkpoint['bert_l_-{}'.format(i)])
        for module in self._modules:
            if 'task' in module:
                self._modules[module].load_state_dict(checkpoint[module+'_state_dict'])



class MLPClassifier(nn.Module):
    """
    Class for Multi-Layer Perceptron Classifier
    """
    def __init__(self, input_dim=768, target_dim=2, hidden_dims=[], nonlinearity=nn.ReLU, dropout=0.0):
        super(MLPClassifier, self).__init__()

        # append input and output dimension to layer list
        hidden_dims.insert(0, input_dim)
        hidden_dims.append(target_dim)

        # stack layers with dropout and specified nonlinearity
        layers = []
        for h, h_next in zip(hidden_dims, hidden_dims[1:]):
            layers.append(nn.Linear(h, h_next))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            if nonlinearity is not None:
                layers.append(parse_nonlinearity(nonlinearity))

        # remove nonlinearity and dropout for output layer
        if nonlinearity is not None:
            layers.pop()
            if dropout > 0:
                layers.pop()

        self.network = nn.Sequential(*layers)

    def forward(self, input):
        output = self.network(input)
        return output


class PrototypeLearner(nn.Module):
    def __init__(self, config, tasks):
        super(PrototypeLearner, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.requires_grad_(False)
        for block in self.encoder.encoder.layer[-(config.unfreeze_num):]:
            for params in block.parameters():
                params.requires_grad = True
        bert_out = 768
        embedding_dim = 500
        # TODO add parameters to make layers maybe

        self.classifier_layer = nn.Sequential(
            nn.Linear(bert_out, embedding_dim),
            nn.ReLU()
        )


    def forward(self, inputs, attention_mask=None):
        encoded = self.encoder(inputs, attention_mask=attention_mask)[0]
        cls_token_enc = encoded[:, 0, :]
        out = self.classifier_layer(cls_token_enc)

        return out

    def calculate_centroids(self, support, num_classes):
        support, support_labels = support
        centroids = torch.zeros((num_classes, 500))
        # compute centroids on support set according to equation 1 in the paper
        unique_labels = support_labels.unique()

        for label in unique_labels:
            centroids[label] = support[(support_labels == label).squeeze(-1)].mean(dim=0)

        return centroids

    def save_model(self, unfreeze_num, snapshot_path):
        # FIXME #2: also save optimizer state_dict, epochs, loss, etc
        # Copy instance of model to avoid mutation while training
        bert_model_copy = deepcopy(self.encoder)

        # Delete frozen layers from model_copy instance, save state_dicts
        state_dicts = {'unfreeze_num': unfreeze_num}
        for module in self._modules:
            if module == 'encoder':
                for i in range(1, unfreeze_num+1):
                    state_dicts['bert_l_-{}'.format(i)] = bert_model_copy.encoder.layer[-i].state_dict()
        state_dicts['outputlayer_state_dict'] = self.classifier_layer.state_dict()
        torch.save(state_dicts, snapshot_path)

    def load_model(self, path, device):
        # Load dictionary with BERT and MLP state_dicts
        checkpoint = torch.load(path, map_location=device)
        unfreeze_num = checkpoint['unfreeze_num']
        # Overwrite last n BERT blocks, overwrite MLP params
        for i in range(1, unfreeze_num + 1):
            self.encoder.encoder.layer[-i].load_state_dict(checkpoint['bert_l_-{}'.format(i)])
        self.classifier_layer.load_state_dict(checkpoint['outputlayer_state_dict'])

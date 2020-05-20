from collections import deque

import torch.nn as nn
from transformers import BertModel
from util import parse_nonlinearity
import torch
from copy import deepcopy

import logging

class Encoder(nn.Module):
    def __init__(self, config, last_linear_layer=False):
        """
        Composed by the BERT base-uncased model to which we attach a set of
        fully-connected layers.
        """
        super(Encoder, self).__init__()
        # BERT
        self.unfreeze_num = config.unfreeze_num
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.requires_grad_(False)
        for block in self.bert.encoder.layer[-(config.unfreeze_num):]:
            for params in block.parameters():
                params.requires_grad = True

        # MLP; layers: linear + dropout (optional) + activation
        bert_cls_token_dims = 768
        hidden_dims = [bert_cls_token_dims] + config.mlp_dims
        layers = []
        for h, h_next in zip(hidden_dims, hidden_dims[1:]):
            layers.append(nn.Linear(h, h_next))
            layers.append(nn.Dropout(p=config.mlp_dropout))
            layers.append(parse_nonlinearity(config.mlp_activation))
        if last_linear_layer:
            layers = layers[:-2]

        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs, attention_mask=None):
        encoded = self.bert(inputs, attention_mask=attention_mask)[0]
        cls_token_enc = encoded[:, 0, :]
        out = self.mlp(cls_token_enc)
        return out

    def get_trainable_params(self):
        # Copy instance of model to avoid mutation while training
        bert_model_copy = deepcopy(self.bert)

        # Delete frozen layers from model_copy instance, save state_dicts
        state_dicts = {'unfreeze_num': self.unfreeze_num}
        for i in range(1, self.unfreeze_num+1):
            state_dicts['bert_l_-{}'.format(i)] = bert_model_copy.encoder.layer[-i].state_dict()
        state_dicts['mlp'] = self.mlp.state_dict()
        return state_dicts

    def load_trainable_params(self, state_dicts):
        unfreeze_num = state_dicts['unfreeze_num']
        # Overwrite last n BERT blocks, overwrite MLP params
        for i in range(1, unfreeze_num + 1):
            self.bert.encoder.layer[-i].load_state_dict(state_dicts['bert_l_-{}'.format(i)])
        self.mlp.load_state_dict(state_dicts['mlp'])


class MultiTaskLearner(nn.Module):
    def __init__(self, config):
        super(MultiTaskLearner, self).__init__()
        self.encoder = Encoder(config)

    def forward(self, inputs, task_name=None, attention_mask=None):
        task_module_name = 'task_{}'.format(task_name)
        assert task_module_name in self._modules

        encoded = self.encoder(inputs, attention_mask=attention_mask)
        classifier = self._modules[task_module_name]
        return classifier(encoded)

    def add_task_classifier(self, task_name, classifier):
        assert issubclass(type(classifier), nn.Module)
        self.add_module('task_{}'.format(task_name), classifier)

    def save_model(self, snapshot_path):
        state_dicts = self.encoder.get_trainable_params()
        for module in self._modules:
            if 'task' in module:
                state_dicts[module+'_state_dict'] = self._modules[module].state_dict()
        torch.save(state_dicts, snapshot_path)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.encoder.load_trainable_params(checkpoint)
        for module in self._modules:
            if 'task' in module:
                self._modules[module].load_state_dict(checkpoint[module+'_state_dict'])


class SLClassifier(nn.Module):
    """
    Class for Single-Layer Classifier
    """
    def __init__(self, input_dim=768, target_dim=2):
        super(SLClassifier, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(input_dim, target_dim)
            )

    def forward(self, input):
        return self.network(input)


class PrototypeLearner(nn.Module):
    def __init__(self, config, input_dim=768, nonlinearity='ReLU', dropout=0.0):
        super(PrototypeLearner, self).__init__()
        self.encoder = Encoder(config, last_linear_layer=True)

    def forward(self, inputs, attention_mask=None):
        encoded = self.encoder(inputs, attention_mask=attention_mask)
        return encoded

    def calculate_centroids(self, support, num_classes):
        support, support_labels = support

        # compute centroids on support set according to equation 1 in the paper
        unique_labels = support_labels.unique()
        centroids = []
        for i in range(num_classes):
            if i in unique_labels:
                centroids.append(support[(support_labels == i).squeeze(-1)].mean(dim=0))
            else:
                # fill centroids for missing labels with random normal noise
                logging.warning('Warning: label not found -> random centroid')
                centroids.append(torch.randn(support.size()[1]).to(support.device))
        return torch.stack(centroids)

    def compute_distance(self, samples, centroids):
        # compute distances
        distances = []
        for i in range(centroids.shape[0]):
            distances.append(torch.norm(samples - centroids[i], dim=1))
        return torch.stack(distances, dim=1)

    def save_model(self, snapshot_path):
        state_dicts = self.encoder.get_trainable_params()
        torch.save(state_dicts, snapshot_path)

    def load_model(self, path, device):
        # Load dictionary with BERT and MLP state_dicts
        checkpoint = torch.load(path, map_location=device)
        self.encoder.load_trainable_params(checkpoint)


class ProtoMAMLLearner(nn.Module):
    def __init__(self, config, input_dim=768, nonlinearity='ReLU', dropout=0.0):
        super(ProtoMAMLLearner, self).__init__()
        self.proto_net = PrototypeLearner(config, input_dim, nonlinearity, dropout)
        self.output_layer = nn.Linear(config.mlp_dims[-1], 2)

    def calculate_output_params(self, prototypes):
        W = 2 * prototypes
        b = - torch.norm(prototypes, p=2, dim=1)
        return W, b

    def initialize_classifier(self, W, b, hard_replace=False):
        # hard replace completely deletes the Parameter from memory
        # if the parameter was specified in the optimizer it needs to be replaced with a parameter
        # before calling optimizer.step()
        if hard_replace:
            del self.output_layer.weight
            del self.output_layer.bias
            self.output_layer.weight = W
            self.output_layer.bias = b
        else:
            self.output_layer.weight.data = W
            self.output_layer.bias.data = b

    def forward(self, inputs, attention_mask=None):
        proto_embedding = self.proto_net(inputs, attention_mask=attention_mask)
        out = self.output_layer(proto_embedding)
        return out

    def save_model(self, snapshot_path):
        # Copy instance of model to avoid mutation while training
        classifier = deepcopy(self.output_layer)

        # Delete frozen layers from model_copy instance, save state_dicts
        state_dicts = self.proto_net.encoder.get_trainable_params()
        state_dicts['output_layer_state_dict'] = classifier.state_dict()

        torch.save(state_dicts, snapshot_path)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.proto_net.encoder.load_trainable_params(checkpoint)
        self.output_layer.load_state_dict(checkpoint['output_layer_state_dict'])

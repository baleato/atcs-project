from collections import deque

import torch.nn as nn
from transformers import BertModel
from util import parse_nonlinearity
import torch


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


class MLPClassifier(nn.Module):
    """
    Class for Multi-Layer Perceptron Classifier
    """
    def __init__(self, input_dim=768, target_dim=2, hidden_dims=None, nonlinearity=None, dropout=0.0):
        super(MLPClassifier, self).__init__()

        # append input and output dimension to layer list
        if hidden_dims is None:
            hidden_dims = []
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
    def __init__(self, config, input_dim=768, target_dim=500, hidden_dims=None, nonlinearity='ReLU', dropout=0.0):
        super(PrototypeLearner, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.requires_grad_(False)
        for block in self.encoder.encoder.layer[-(config.unfreeze_num):]:
            for params in block.parameters():
                params.requires_grad = True

        self.classifier_layer = MLPClassifier(input_dim, target_dim, hidden_dims, nonlinearity, dropout)


    def forward(self, inputs, attention_mask=None):
        encoded = self.encoder(inputs, attention_mask=attention_mask)[0]
        cls_token_enc = encoded[:, 0, :]
        out = self.classifier_layer(cls_token_enc)

        return out

    def calculate_centroids(self, support, num_classes): #, query_labels#, train_iter, task):
        support, support_labels = support
        # compute centroids on support set according to equation 1 in the paper
        unique_labels = support_labels.unique()
        centroids = []
        for i in range(num_classes):
            if i in unique_labels:
                centroids.append(support[(support_labels == i).squeeze(-1)].mean(dim=0))
            else:
                # fill centroids for missing labels with random normal noise
                centroids.append(torch.randn(500))
        return torch.stack(centroids)


class ProtoMAMLLearner(nn.Module):
    def __init__(self, config, input_dim=768, target_dim=500, hidden_dims=None, nonlinearity='ReLU', dropout=0.0):
        super(ProtoMAMLLearner, self).__init__()
        self.proto_net = PrototypeLearner(config, input_dim, target_dim, hidden_dims, nonlinearity, dropout)
        self.output_layer = nn.Linear(target_dim, 2)

    def calculate_output_params(self, prototypes):
        W = nn.Parameter(2 * prototypes)
        b = nn.Parameter(- torch.norm(prototypes, p=2, dim=1))
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

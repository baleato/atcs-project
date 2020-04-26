import torch.nn as nn
from transformers import BertModel
import torch


class MetaLearner(nn.Module):

    def __init__(self, config):
        super(MetaLearner, self).__init__()
        # TODO: include BERT
        # TODO: define classifier
        self.classifier

    def forward(self, inputs):
        # TODO: use BERT
        return self.classifier(inputs)


class MLPClassifier(nn.Module):
    """
    Class for Multi-Layer Perceptron Classifier
    """
    def __init__(self, input_dim, target_dim):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, target_dim)
        )

    def forward(self, input):
        output = self.network(input)
        output = torch.mean(output, dim=1)
        return output

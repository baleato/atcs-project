import torch.nn as nn
from transformers import BertModel


class MetaLearner(nn.Module):

    def __init__(self, config):
        super(MetaLearner, self).__init__()
        # TODO: include BERT
        # TODO: define classifier
        self.classifier

    def forward(self, inputs):
        # TODO: use BERT
        return self.classifier(inputs)

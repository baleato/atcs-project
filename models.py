from collections import deque

import torch.nn as nn
from transformers import BertModel
from util import parse_nonlinearity


class MetaLearner(nn.Module):
    def __init__(self, config, n_classes=11):
        super(MetaLearner, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.requires_grad_(False)
        # TODO: unfreeze top n layers
        for block in self.encoder.encoder.layer[-(config.unfreeze_num):]:
            for params in block.parameters():
                params.requires_grad = True



        # top_n_bert_layers = deque(
        #    self.encoder.parameters(),
        #    maxlen=config.n_layers_bert_trained)
        # for params in top_n_bert_layers:
        #   params.requires_grad = True
        n_bert_embed = 768
        self.emo_classifier = MLPClassifier(n_bert_embed, n_classes)

    def forward(self, sentences, attention_mask=None):
        encoded = self.encoder(sentences, attention_mask=attention_mask)[0]
        cls_token_enc = encoded[:, 0, :]
        return self.emo_classifier(cls_token_enc)


class MLPClassifier(nn.Module):
    """
    Class for Multi-Layer Perceptron Classifier
    """
    def __init__(self, input_dim, target_dim, hidden_dims=[], nonlinearity=None, dropout=0.0):
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

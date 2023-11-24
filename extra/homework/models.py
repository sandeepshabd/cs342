import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
import numpy as np


class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """

        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
    """
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, downsample=True):
            """
            Your code here.
            Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
            Optionally, repeat this pattern a few times and add in a residual block
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            super().__init__()
            self.conv1 = self._create_causal_conv(in_channels, out_channels, kernel_size, dilation, dropout)
            self.conv2 = self._create_causal_conv(out_channels, out_channels, kernel_size, dilation, dropout)

            # Downsample if necessary
            self.down = nn.Conv1d(in_channels, out_channels, 1) if downsample else None
        def _create_causal_conv(self, in_channels, out_channels, kernel_size, dilation, dropout):
            """
            Create a single causal convolution layer followed by ReLU and dropout.
            """
            padding = (kernel_size - 1) * dilation  # Padding for causality
            return nn.Sequential(
                nn.ConstantPad1d((padding, 0), 0),
                nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )


        def forward(self, x):
            out = self.conv2(self.conv1(x))
            residual = x if self.down is None else self.down(x)
            return out + residual

    #def __init__(self):
        """
        Your code here

        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
    def __init__(self, num_layers=8, num_channels=50, vocab_size=28, kernel_size=3, dropout=0.05):
        super().__init__()
        self.layers = nn.ModuleList()
        self.first_char = nn.Parameter(torch.rand(vocab_size, 1), requires_grad=True)
        
        in_channels = vocab_size
        dilation_size = 1
        for _ in range(num_layers):
            self.layers.append(self.CausalConv1dBlock(in_channels, num_channels, kernel_size, dilation_size, dropout))
            dilation_size *= 2
            in_channels = num_channels
        
        self.classifier = nn.Conv1d(num_channels, vocab_size, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_first_char = self.first_char.expand(x.size(0), -1, -1)
        if x.shape[2] == 0:
            return self.log_softmax(batch_first_char)

        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.classifier(out)
        out = torch.cat([batch_first_char, out], dim=2)
        return self.log_softmax(out)

    def predict_all(self, some_text):
        one_hot = utils.one_hot(some_text)
        p = self.forward(one_hot.unsqueeze(0))
        return p.squeeze(0)


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r

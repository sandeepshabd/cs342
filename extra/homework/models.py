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
        def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
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
            self.padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                   padding=self.padding, dilation=dilation)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                                   padding=self.padding, dilation=dilation)
            self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None


        def forward(self, x):
            out = self.relu(self.conv1(x))
            out = self.conv2(out)[:, :, :-self.padding]  # Remove extra padding for causality
            residual = x if self.residual is None else self.residual(x)
            return self.relu(out + residual[:, :, :out.size(2)])

    #def __init__(self):
        """
        Your code here

        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
    def __init__(self, vocab_size, num_channels, num_blocks=2, kernel_size=3, dilation_factor=1):

        super().__init__()
        layers = []
        for i in range(num_blocks):
            dilation = dilation_factor ** i
            in_channels = vocab_size if i == 0 else num_channels
            layers.append(self.CausalConv1dBlock(in_channels, num_channels, kernel_size, dilation))
        
        self.tcn = nn.Sequential(*layers)
        self.first_char_prob = nn.Parameter(torch.randn(vocab_size))
        self.output_layer = nn.Linear(num_channels, vocab_size)

    def forward(self, x):
        """
        Your code here
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        out = self.tcn(x)
        out = self.output_layer(out.transpose(1, 2))
        out = F.pad(out, (1, 0), "constant", 0)  # Shift for predicting next character
        return out

    def predict_all(self, some_text):
        """
        Your code here

        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        x = self.text_to_one_hot(some_text)
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1)


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r

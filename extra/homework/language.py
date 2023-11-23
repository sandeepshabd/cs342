import string
import torch
from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    
    all_probs = model.predict_all(some_text)
    likelihood = 0
    for i in range(len(some_text)):
        likelihood += all_probs[utils.vocab.find(some_text[i]), i]
    return likelihood
"""
    # Initialize log-likelihood
    log_likelihood = 0.0

    # Iterate over the text to compute the sum of log-probabilities of the actual next characters
    for i in range(len(some_text)+1):
        # The substring up to the current character
        print("i",i)
        substring = some_text[:i]
        
        # Predict the log-probabilities for the next character
        log_probs_next_char = model.predict_next(substring)
        

        
        # Get the log-probability for the actual next character
        log_prob_actual_next_char = log_probs_next_char[i]
        
        # Add this log-probability to the total log-likelihood
        log_likelihood += log_prob_actual_next_char

    return log_likelihood.item()  # Convert from tensor to Python float

"""

def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    sentence = ''  # Initialize the sentence as an empty string
    vocab = string.ascii_lowercase + ' .'
    for _ in range(max_length):
        # Predict the log-probabilities of the next character using the current sentence
        log_probs = model.predict_next(sentence)
        
        # Convert log probabilities to actual probabilities
        probs = torch.exp(log_probs)
        
        # Sample a character from the probability distribution
        char_index = torch.multinomial(probs, 1).item()
        
        # Assume we have a function utils.index_to_char to convert an index to a character
        char = vocab[char_index]
        
        # Append the sampled character to the sentence
        sentence += char
        
        # Terminate if the period is reached
        if char == '.':
            break
            
    return sentence


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    heap = TopNHeap(N=beam_size)
    next_likes = model.predict_next('')
    
    for next_step, next_like in enumerate(next_likes):
        element = ((next_like).item(), utils.vocab[next_step])
        if element not in [el[1] for el in heap.elements]:
            heap.add(element)

    for _ in range(max_length):        
        for seq, score in heap.elements:
            if score[-1] == '.':
                continue
            next_likes = model.predict_next(score[-1])
            
            
            for next_step, next_like in enumerate(next_likes):
                if average_log_likelihood:
                    curr_len = len(score)
                    avg_like = (seq*curr_len + next_like.item())/(curr_len + 1)
                    element = (avg_like, score + utils.vocab[next_step])
                else:
                    element = (seq + next_like.item(), score + utils.vocab[next_step])            
                if element[1] not in [el[1] for el in heap.elements]:
                    heap.add(element)

    sorted_heap = sorted(heap.elements, key=lambda x: x[0])[::-1]
    output = [el[1] for el in sorted_heap[:n_results]]
    return output



if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))

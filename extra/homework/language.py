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
    # Initialize log-likelihood
    log_likelihood = 0.0

    # Iterate over the text to compute the sum of log-probabilities of the actual next characters
    for i in range(len(some_text)):
        # The substring up to the current character
        substring = some_text[:i]
        
        # Predict the log-probabilities for the next character
        log_probs_next_char = model.predict_next(substring)
        

        
        # Get the log-probability for the actual next character
        log_prob_actual_next_char = log_probs_next_char[i]
        
        # Add this log-probability to the total log-likelihood
        log_likelihood += log_prob_actual_next_char

    return log_likelihood.item()  # Convert from tensor to Python float



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
    
    for _ in range(max_length):
        # Predict the log-probabilities of the next character using the current sentence
        log_probs = model.predict_next(sentence)
        
        # Convert log probabilities to actual probabilities
        probs = torch.exp(log_probs)
        
        # Sample a character from the probability distribution
        char_index = torch.multinomial(probs, 1).item()
        
        # Assume we have a function utils.index_to_char to convert an index to a character
        char = utils.vocab(char_index)
        
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
    beams = [("", 0)]

    for _ in range(max_length):
        candidates = []
        
        # Expand each beam
        for sentence, log_likelihood in beams:
            # Stop expanding this beam if it ends with a period
            if sentence.endswith('.'):
                candidates.append((sentence, log_likelihood))
                continue

            # Predict the next character's log-probabilities
            log_probs_next = model.predict_next(sentence)

            # Consider top `beam_size` next characters
            topk_log_probs, topk_indices = torch.topk(log_probs_next, beam_size)

            for log_prob, index in zip(topk_log_probs, topk_indices):
                next_char = utils.vocab(index.item())
                new_sentence = sentence + next_char
                new_log_likelihood = log_likelihood + log_prob.item()
                if average_log_likelihood:
                    new_log_likelihood /= len(new_sentence)
                candidates.append((new_sentence, new_log_likelihood))

        # Keep top `beam_size` beams
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    # Pick the top `n_results` unique sentences
    unique_sentences = set()
    results = []
    for sentence, _ in beams:
        if sentence not in unique_sentences and len(results) < n_results:
            results.append(sentence)
            unique_sentences.add(sentence)

    return results


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

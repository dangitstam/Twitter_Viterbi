from collections import Counter
from math import log
import numpy as np


class NGramModel:
    """
    Generalized ngram model.

    ex. MLE for trigrams:
      P(x3 | x1 x2) = c(x1 x2 x3) / c(x1 x2)
    
    For order-1 ngram models, counts are stored as words
    themselves to their frequency instead of tuples.
    """

    def __init__(self, order):
        self.order = order

        # This is to be used for unigram models, where the MLE is simply
        # the frequency of the token in the corpus.
        self._total_token_count = 0
        self._ngram_frequencies = Counter()
        self._context_frequencies = Counter()

    def update(self, input_tokens):
        """
        Given a training example, updates the ngram model by collecting
        all ngrams and (n - 1) grams in the example.
        """

        # Example is already a list
        ngrams = list(zip(*[input_tokens[i:] for i in range(self.order)]))
        contexts = list(zip(*[input_tokens[i:] for i in range(self.order - 1)]))

        self._ngram_frequencies.update(ngrams)
        self._context_frequencies.update(contexts)

    def maximum_likelihood_estimate(self, input_tokens):
        """
        Maximum Likelihood Estimate for ngrams.
        
        For a given ngram returns the count of the n-length sequence divided
        by the count of the (n - 1) length context. For order n = 1, context
        count is then the total number of words seen in training.

        Parameters
        ----------
        ngram : A tuple of integers representing an ngram.
        """

        if input_tokens in self._ngram_frequencies:
            ngram_count = self._ngram_frequencies[input_tokens]

            # MLE for orders > 1
            if self.order > 1 and input_tokens[:-1] in self._context_frequencies:
                context = input_tokens[:-1]
                context_count = self._context_frequencies[context]
                return ngram_count / context_count
            else:
                # Special case for unigrams; MLE is the frequency of the
                # word in the corpus.
                return ngram_count / self._total_token_count

        return 0

    def get_ngram_frequencies(self):
        return dict(self._ngram_frequencies)

    def get_context_frequencies(self):
        return dict(self._context_frequencies)

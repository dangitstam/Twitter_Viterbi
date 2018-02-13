from collections import Counter
from math import log

from NGramModel import NGramModel
from symbols import STOP, UNKNOWN
from vocab import sieve_by_threshold


class InterpolatedNGramModel:
    """ Implementation of an NGram Model using Linear Interpolation
        for smoothing. """

    def __init__(self, corpus, order, sieve=2):
        vocab, M, U = sieve_by_threshold(corpus, sieve)
        self.models = []

        # Populate the interploated model with NGram models with orders
        # 1 through N
        for i in range(order):
            ngram_freq, context_freq, ith_vocab = NGramModel.generate_counts(corpus, vocab, i + 1)
            self.models.append(NGramModel(ngram_freq, context_freq, ith_vocab, M, i + 1))
            self.vocab = ith_vocab

        self.M = M
        self.U = U
        self.order = order
        self.START = NGramModel.generate_start_symbol(self.order)


    def QML(self, ngram, lambdas):
        # Interpolated MLEs across all order-1 ... order-n models.

        interpolated_probability = 0
        for i, l in enumerate(lambdas):
            offset = self.order - 1 - i
            interpolated_probability += (l * self.models[i].QML(ngram[offset:]))

        return interpolated_probability


    def log2_probability(self, si, lambdas):
        # Returns the log base 2 probability of a sentence 'si'.
        # Assumes 'si' is a space-separated sentence.

        si = self.START + " " + si + " " + STOP
        si = [w if w in self.vocab else UNKNOWN for w in si.split()]
        ngrams = list(zip(*[si[i:] for i in range(self.order)]))

        log2_probability = 0
        for ngram in ngrams:
            ngram_probability = self.QML(ngram, lambdas)
            log2_probability += log(ngram_probability, 2)

        return log2_probability

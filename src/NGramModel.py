
from collections import Counter
from math import log

from symbols import START, STOP, UNKNOWN

class NGramModel:
    """ Generalized NGram Model. """

    def __init__(self, ngram_frequencies, context_frequencies, vocab, M, order):
        # NGram models involving storing counts of length n and
        # length (n - 1) sequences:
        #
        # ex. MLE for trigrams:
        #   q_ML(x3 | x1 x2) = c(x1 x2 x3) / c (x1 x2)
        #
        # For order 1 ngram models, counts are stored as words
        # themselves to their frequency instead of tuples.
        self.ngram_frequencies = ngram_frequencies
        self.context_frequencies = context_frequencies
        self.vocab = vocab
        self.M = M
        self.order = order
        self.START = NGramModel.generate_start_symbol(order)


    def QML(self, ngram):
        # Maximum Likelihood Estmate for ngrams: For a given ngram
        # returns the count of the n-length sequence divided by the
        # count of the (n - 1) length context.
        #
        # For order n = 1, context count is then the total number of
        # words seen in training.
        
        if ngram in self.ngram_frequencies:
            ngram_count = self.ngram_frequencies[ngram]

            # MLE for orders > 1
            if self.order > 1 and ngram[:-1] in self.context_frequencies:
                context = ngram[:-1]
                context_count = self.context_frequencies[context]
                return ngram_count / context_count
            else:
                # Special case for unigrams; MLE is the frequency of the
                # word in the corpus.
                return ngram_count / self.M

        return 0


    def log2_probability(self, si):
        # Calculates log base 2 probability of the sentence.
        #
        # Assumes si has no special symbols present.
        si = self.START + " " + si + " " + STOP
        si = [w if w in self.vocab else UNKNOWN for w in si.split()]
        ngrams = list(zip(*[si[i:] for i in range(self.order)]))

        log2_probability = 0
        for ngram in ngrams:
            ngram_probability = self.QML(ngram)

            # None serves as a flag for negative infinity.
            if ngram_probability == 0.0:
                return None

            log2_probability += log(ngram_probability, 2)

        return log2_probability


    @staticmethod
    def generate_counts(corpus, vocab, order):
        # Given a corpus represented as a list of sentences, generates
        # the n-length and n-minus-one-length counts.
        #
        # In the event that order = 1, contexts and context_frequencies
        # will be empty.
        #
        # Zip idea courtesy of
        # locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
        # 
        # Assumes vocab has been populated with special start, stop, and unknown
        # symbols.

        # Start symbols allow for easier generalization across orders.
        # START is unique for every order.
        START = NGramModel.generate_start_symbol(order)

        # Add start and stop symbols to vocabulary.
        vocab.update(START.split())
        vocab.add(STOP)

        ngrams = []
        contexts = []
        M = 0
        for si in corpus:
            # Append starts and stop, replace unknowns with unknown tag.
            si = START + " " + si + " " + STOP
            si = [w if w in vocab else UNKNOWN for w in si.split()]

            # Generate counts of C(x_1 ... x_n) and C(x_1 ... x_(n - 1))
            ngrams += list(zip(*[si[i:] for i in range(order)]))
            contexts += list(zip(*[si[i:] for i in range(order - 1)]))


        # Maps ngrams and contexts to their frequencies
        ngram_frequencies = Counter(ngrams)
        context_frequencies = Counter(contexts)

        return dict(ngram_frequencies), dict(context_frequencies), vocab


    @staticmethod
    def generate_start_symbol(order):
        return ' '.join([START for i in range(order - 1)])









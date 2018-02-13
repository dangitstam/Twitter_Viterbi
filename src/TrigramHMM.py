
from collections import Counter, defaultdict as D

from symbols import *
from NGramModel import NGramModel
from InterpolatedNGramModel import InterpolatedNGramModel
from vocab import sieve_by_threshold


class HMM:
    
    def __init__(self, order, lambdas):
        self.order = order
        self.lambdas = lambdas
        self.transitions = None
        self.emissions_cnts = D(lambda: 0)  # Counts of xi -> yi
        self.src_word_cnts = D(lambda: 0)  # Counts of xi

        # Vocab for emissions
        self.emissions_vocab = None
        self.states = None


    def q(self, ngram):
        return self.transitions.QML(ngram, self.lambdas)


    def emission_probability(self, xi, yi):
        # 'UNK' low-frequency words using twitter heuristics.
        # If the word could not be unked and is not present
        # in the vocabulary, default to <UNKNOWN>

        # Trick allows UNKs to pass through unchanged in pre-processing.
        if xi not in UNKS:
            xi = twitter_unk(xi, self.emissions_vocab)

        return self.emissions_cnts[(xi, yi)] / self.src_word_cnts[xi]


    def generate_emission_mles(self, word_mappings):

        # Update c(xi -> yi) and c(xi)
        for wm in word_mappings:
            xi, yi = wm

            # 'UNK' low-frequency words using twitter heuristics.
            xi = twitter_unk(xi, self.emissions_vocab)

            self.emissions_cnts[(xi, yi)] += 1  
            self.src_word_cnts[xi] += 1


    def train(self, examples):
        # Given a corpus of the form [[x1, y1] ... [xn, yn], trains a
        # Hidden Markov Model, collecting counts for the emission and
        # transition MLEs.

        # Sieve out the tags from every sentence in the corpus, create
        # space-separated sentences with them for training the NGramModel
        # that calculates q(yi | ...) frequencies.

        STARTS = NGramModel.generate_start_symbol(self.order)
        def add_starts_and_stops(s):
            return STARTS + " " + s + " " + STOP

        transition_train = [add_starts_and_stops(" ".join([xi[1] for xi in x])) for x in examples]

        # Get emission words for unking
        word_freq = Counter()
        sentences = sum([[[xi[0] for xi in x] for x in examples]], [])

        # maps are lazy, list consumes calculations.
        list(map(lambda xs: word_freq.update(xs), sentences))

        # Create a vocabulary for emitted words out of words that appear
        # at least twice.
        self.emissions_vocab = set([w for w, f in word_freq.items() if f >= 2])
        self.total_vocab = [w for w, f in word_freq.items()]

        # Keeps all words when sieve = 0
        V, M, U = sieve_by_threshold(transition_train, 0)

        self.states = V
        self.transitions = InterpolatedNGramModel(transition_train, self.order, sieve=0)

        # Calculate emission MLEs provided by the corpus.
        list(map(self.generate_emission_mles, examples))


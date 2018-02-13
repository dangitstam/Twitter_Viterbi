
from collections import Counter
import math
import operator

from symbols import STOP, UNKNOWN

#### Utility functions for sieving vocabulary ####

def M(corpus):
    # Given a list of sentences as a corpus, return the total number
    # of words (including duplicates).

    M = 0
    for line in corpus:
        # +1 for STOP symbol.
        M += (len(line.split()) + 1)

    return M

def U(corpus):
    return len(Counter(corpus.split()))


def count_frequencies(corpus):
    # Maps words to their frequencies given a corpus (list of sentences).
    # Includes obligatory stops in M.
    #
    # Returns mappings, number of total words seen and number of different
    # words seen.

    M = 0  # Total words
    U = 0  # Number of different words

    bag_of_words = []
    for line in corpus:
        tokens = line.split()
        bag_of_words += tokens

        # +1 accounts for stops that are added later.
        M += (len(tokens) + 1)

    word_to_freq = dict(Counter(bag_of_words))
    U = len(word_to_freq)
    return word_to_freq, M, U


#### Methods for selecting most frequent words in a vocabulary ####

def sieve_by_threshold(corpus, th):
    # Returns a set of words from 'corpus' that appeared at least
    # 'th' times, along with total words seen and total different words seen.

    word_to_freq, M, U = count_frequencies(corpus)

    # Sieves a dictionary by leaving out all words that appear < th times.
    above_threshold =  set([w for w, f in word_to_freq.items() if f >= th])

    return above_threshold, M, U


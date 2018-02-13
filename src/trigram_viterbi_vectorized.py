
import itertools
import json
from math import log
import operator
import sys
import time

import numpy as np

from NGramModel import NGramModel
from symbols import *
from TrigramHMM import HMM
from util import expected_vs_actual


def main():

    if len(sys.argv) < 3:
        print("Usage: python scratch.py <training> <validation>")
        sys.exit()


    # Collect training and validation data.
    training_data = []
    with open(sys.argv[1], 'r') as f:
        training_data = [json.loads(e) for e in f.readlines()]

    validation = []
    with open(sys.argv[2], 'r') as f:
        validation = [json.loads(e) for e in f.readlines()]


    # Train the model.
    l1 = 0.01
    l2 = 0.2917
    l3 = 1 - l1 - l2
    lambdas = [l1, l2, l3]

    hmm = HMM(3, lambdas)
    hmm.train(training_data)

    states = sorted(list(hmm.states))
    state_to_idx = {}
    for i, st in enumerate(states):
        state_to_idx[st] = i

    N = len(states)

    #### Precompute tranisitions and emissions for vectorized Viterbi ####
    # Constrict a transition matrix over all possible triplets
    # of states.
    #
    # Access by [v][u][w] where w is yn - 2, u is yn - 1.
    transition_matrix = np.zeros((N, N, N), dtype=np.float64)
    for i, w in enumerate(states):
        for j, u in enumerate(states):
            for k, v in enumerate(states):
                transition_matrix[k][j][i] = hmm.q((w, u, v))

    transition_matrix = np.log2(transition_matrix)  # Log only needed once!


    # Construct an emission matrix over the entire corpus
    emission_matrix = {}
    def emission_entry(word, word_to_emissions):

        emissions = np.zeros(N)
        for i, s in enumerate(states):

            # Log only needed once!
            emissions[i] = np.log2(hmm.emission_probability(word, s))

        word_to_emissions[word] = emissions

    for word in hmm.total_vocab:  # Emissions for each word in the corpus
        word = twitter_unk(word, hmm.emissions_vocab)
        emission_entry(word, emission_matrix)


    # Artificial emissions for stop allows the forcing
    # of stop to be the yn+1st state (one hot encoding).
    stop_emissions = np.zeros(N)
    stop_emissions[state_to_idx[STOP]] = 1
    emission_matrix[STOP] = stop_emissions


    # Iterate over the validation set, calculating raw accuracy.
    tp, total = 0, 0
    start = time.clock()
    progress = "\rIteration: {} Accuracy: {} "
    for i, ex in enumerate(validation):
        words = " ".join([w[0] for w in ex])
        labels = " ".join([w[1] for w in ex])

        path = viterbi_pi(words, states, state_to_idx, transition_matrix, emission_matrix, hmm)

        path_tp, path_total = expected_vs_actual(path, labels)
        tp += path_tp
        total += path_total

        print(progress.format(i, tp / total), end="")
        sys.stdout.flush()


    duration = time.clock() - start
    print()
    print("Time to label validation:", duration)
    print("Raw accuracy:", tp / total)


def viterbi_pi(x, states, state_to_idx, transition_matrix, emission_matrix, hmm):

    # UNK words of the observed states.
    x = [twitter_unk(xi, hmm.emissions_vocab) for xi in x.split(" ")]

    # For single word tweets; return the state that maximizes emission.  
    if len(x) == 1:
        max_label = None
        max_likelihood = float('-inf')
        for i, st in enumerate(states):
            candidate = hmm.emission_probability(x[0], st)
            if candidate > max_likelihood:
                max_label = st

        return st

    # Over all possible paths and all possible states
    # V serves as the pi table, being indexed first by position of
    # the sentence, then by yi - 2, and then by yi - 1.
    #
    # V is defined in log space, -inf for 0 entries and 0
    # for pi(0, START, START), enforcing that all paths
    # start with START START.
    K = len(states)
    n = len(x)
    V = np.zeros((K, K), dtype=np.float64)
    V[V == 0] = float('-inf')
    V[state_to_idx[START]][state_to_idx[START]] = 0

    # Stores each w that lead to the highest scoring ending of (u, v)
    # Offset by 2 allows for the kth indices to depend on the (k + 2)th
    # entry.
    P = np.zeros((n + 2, K, K), dtype=np.int64)

    # For each word in the sentence
    for k, xk in enumerate(x):
        # No emission needed as START is taken of first
        r = transition_matrix + V

        # Update backpointers
        P[k] = r.argmax(axis=-1)

        # Add emissions after each iteration so the next pi has them.
        V = np.amax(r, axis=-1)
        V = (V.T + emission_matrix[xk]).T


    # Handle STOP with an obligatory transition to STOP.
    final = transition_matrix + (V.T + emission_matrix[STOP]).T

    # Indicies of maximal yn-1, yn, stop.
    stop, yn, yn_minus_1 = np.unravel_index(np.argmax(final), final.shape)
    output_idxs = [yn_minus_1, yn]

    # Backtrace
    for k in range(len(x) - 2, 0, -1):

        # P follows the same convention as transition since
        # each subsequent PI is built off of the transition matrix.
        yk = P[k + 1][output_idxs[1]][output_idxs[0]]
        output_idxs.insert(0, yk)

    return " ".join([states[idx] for idx in output_idxs])




if __name__ == "__main__":
    main()

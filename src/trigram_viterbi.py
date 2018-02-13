
import json
from math import log
import operator
import sys
import time

import numpy as np

from NGramModel import NGramModel
from symbols import START, STOP, twitter_unk
from TrigramHMM import HMM
from util import expected_vs_actual


def main():

    if len(sys.argv) < 3:
        print("Usage: python scratch.py <training> <dev> <show_output>")
        sys.exit()

    training_data = []
    with open(sys.argv[1], 'r') as f:
        training_data = [json.loads(e) for e in f.readlines()]

    dev = []
    with open(sys.argv[2], 'r') as f:
        dev = [json.loads(e) for e in f.readlines()]


    show_output = False
    if len(sys.argv) > 3:
        show_output = True


    l1 = 0.01
    l2 = 0.2917
    l3 = 1 - l1 - l2
    lambdas = [l1, l2, l3]

    hmm = HMM(3, lambdas)
    hmm.train(training_data)


    # Iterating over dev:
    tp, total = 0, 0
    start = time.clock()
    progress = "\rIteration: {}"
    for i, ex in enumerate(dev):
        words = " ".join([w[0] for w in ex])
        labels = " ".join([w[1] for w in ex])

        path = viterbi_pi(words, hmm)
        if show_output:
            print(words)
            print("\tViterbi:", path)
            print("\tActual: ", labels)

        path_tp, path_total = expected_vs_actual(path, labels)
        tp += path_tp
        total += path_total

        print(progress.format(i), end="")
        sys.stdout.flush()


    duration = time.clock() - start
    print()
    print("Time to label development:", duration)
    print("Raw accuracy:", tp / total)


def viterbi_pi(x, hmm):

    # Establish indices for states and initialize V's first column.
    # START and STOP should only be considered at the ends of the sentence.
    states = sorted(list(hmm.states))

    # UNK words of the observed states.
    x = [twitter_unk(xi, hmm.emissions_vocab) for xi in x.split(" ")]

    # Single word tweets
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
    K = len(states)
    n = len(x)
    V = np.zeros((n + 1, K, K))  # n + 1 accounts for starts?
    V[V == 0] = float('-inf')
    V[0] = 1  # Handles starts

    # Stores each w that lead to the highest scoring ending of (u, v)
    P = np.zeros((n + 2, K, K), dtype=np.int64)

    for idx, xk in enumerate(x):  # For each word in the sentence

        k = idx + 1  # V[0] init. with start probabilities.
        for u, su in enumerate(states):
            for v, sv in enumerate(states):

                emission = hmm.emission_probability(xk, sv)
                max_w, max_w_idx = float('-inf'), 0

                # Probabilities are always 0 if the emission likelihood is 0.
                if emission > 0.0:
                    for w, sw in enumerate(states):
                        transition = hmm.q((sw, su, sv))
                        candidate = V[k - 1][w][u] + np.log2(transition) + np.log2(emission)

                        # Current (sw, su, sv) candidate for best wth word to precede
                        # uth and vth words.
                        if max_w < candidate:
                            max_w = candidate
                            max_w_idx = w

                V[k][u][v] = max_w
                P[k][u][v] = max_w_idx


    # Handle STOP
    yi_minus_1, yi = None, None
    max_final = float('-inf')
    for u, su in enumerate(states):
        for v, sv in enumerate(states):
            max_w, max_w_idx = float('-inf'), 0

            transition = hmm.q((su, sv, STOP))
            candidate = V[-1][u][v] + np.log2(transition)
            if max_final < candidate:
                max_final = candidate
                yi_minus_1 = u
                yi = v


    ########### Backtrace #########
    output_idxs = [yi_minus_1, yi]


    # -2 since first two are taken care of
    for k in range(len(x) - 2, 0, -1):
        yk = P[k + 2][output_idxs[0]][output_idxs[1]]
        output_idxs.insert(0, yk)


    return " ".join([states[idx] for idx in output_idxs])





if __name__ == "__main__":
    main()

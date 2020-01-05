import numpy as np
from viterbi.models.ngram_model import NGramModel


class HiddenMarkovModel:
    def __init__(
        self,
        vocab,
        order=3,
        token_namespace="tokens",
        label_namespace="labels",
        start_token="@@START@@",
        end_token="@@END@@",
    ):
        self.vocab = vocab
        self.order = order
        self.token_namespace = token_namespace
        self.label_namespace = label_namespace
        self.start_token, self.end_token = start_token, end_token
        self.start_token_id = vocab.get_token_index(start_token, label_namespace)
        self.end_token_id = vocab.get_token_index(end_token, label_namespace)

        token_namespace_size = vocab.get_vocab_size(token_namespace)
        label_namespace_size = vocab.get_vocab_size(label_namespace)

        # A token_namespace x label_namespace containing emission probabilities.
        self.emission_matrix = np.zeros((token_namespace_size, label_namespace_size))

        # A label_namespace ^ order sized matrix containing transition probabilities.
        # For a trigram HMM, transition_matrix[w][u][v] = P(V | w, u).
        self.transition_matrix = np.zeros((label_namespace_size,) * order)

        # An ngram model to help construct the transition matrix over labels.
        self._label_ngram_model = NGramModel(order)

        # Store the frequencies of seen labels here. Set the start and end token
        # frequencies artificially by 1 to avoid division by zero (they will
        # never be seen during training).
        self._label_count = np.zeros((label_namespace_size,), dtype=int)
        self._label_count[self.start_token_id] = 1
        self._label_count[self.end_token_id] = 1

    def train(self, dataset):
        """
        TODO: dataset should be an iterator over the corpus
        """
        for instance in dataset:
            tokens = instance["token_ids"]
            labels = instance["label_ids"]

            # Update emissions matrix and label counts to contain the raw counts of each
            # token-label and labeloccurrence. Later, each value is divided by the label occurrence.
            for token, label in zip(tokens, labels):
                self.emission_matrix[token][label] += 1
                self._label_count[label] += 1

            # Update the ngram model. Each series of labels is prepended with
            # (order - 1) start tokens and appeneded with one end token.
            labels = (
                [self.start_token_id] * (self.order - 1) + labels + [self.end_token_id]
            )
            self._label_ngram_model.update(labels)

        self._construct_emission_matrix()
        self._construct_transition_matrix()

    def _construct_emission_matrix(self):
        self.emission_matrix /= self._label_count

    def _construct_transition_matrix(self):
        """
        Given the current state of the ngram model, construct a transition
        matrix such that each entry represents the maximum likelihood estimate
        of its index.

        Ex. For a trigram HMM, a transition matrix Q takes the form
                Q[W][U][V] = q(v | w, u)
            where q(v | w,  u) = c(w,  u, v) / c(w, u)
        """
        all_observed_ngrams = set(
            self._label_ngram_model.get_ngram_frequencies().keys()
        )
        for ngram in all_observed_ngrams:
            self.transition_matrix[ngram] = self._label_ngram_model.maximum_likelihood_estimate(ngram)

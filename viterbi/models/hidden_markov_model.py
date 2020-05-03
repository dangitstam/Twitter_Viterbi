import itertools

import numpy as np
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from tqdm import tqdm

from viterbi.data.util import DEFAULT_START_TOKEN, DEFAULT_END_TOKEN
from viterbi.models.ngram_model import NGramModel


class HiddenMarkovModel:
    def __init__(
        self,
        vocab,
        order=3,
        token_namespace="tokens",
        label_namespace="labels",
        start_token=DEFAULT_START_TOKEN,
        end_token=DEFAULT_END_TOKEN,
        language_model=MLE
    ):
        self.vocab = vocab
        self.order = order
        self.token_namespace = token_namespace
        self.label_namespace = label_namespace
        self.start_token = start_token
        self.end_token = end_token
        self.language_model = language_model

        # Infer start and end token IDs.
        self.start_token_id = vocab.get_token_index(start_token, label_namespace)
        self.end_token_id = vocab.get_token_index(end_token, label_namespace)

        # Init. transition and emission matrices.
        self._init_parameters()

        # A flag to check if the model has been trained.
        self._is_trained = False

    def _init_parameters(self):
        token_namespace_size = self.vocab.get_vocab_size(self.token_namespace)
        label_namespace_size = self.vocab.get_vocab_size(self.label_namespace)

        # A token_namespace x label_namespace containing emission probabilities.
        self.emission_matrix = np.zeros((token_namespace_size, label_namespace_size))

        # A label_namespace ^ order sized matrix containing transition probabilities.
        # For a trigram HMM, transition_matrix[w][u][v] = P(v | w, u).
        self.transition_matrix = np.zeros((label_namespace_size,) * self.order)

        # An ngram model to help construct the transition matrix over labels.
        self._label_ngram_model = NGramModel(self.order)

        # TODO: This will eventually replace the current ngram model.
        # TODO: Unit tests pass! Make this toggleable!
        self._lm_ngram_model = self.language_model(self.order)

        # A flag to check if the model has been trained.
        self._is_trained = False

    def train(self, dataset):
        """
        Initializes and trains HMM parameters.

        For HMMs that are already trained, re-training on the given dataset will destroy and
        re-initialize, and re-learn the HMM's parameters.  

        Parameters
        ----------
        dataset :
            An iterator over the dataset. Assumes each example has a "token_ids" mapped to a
            List[int] of input tokens, and a "label_ids" mapped to a List[int] of labels.
        """

        if self._is_trained:
            self._init_parameters()

        train_text = []
        for instance in dataset:
            token_ids = instance["token_ids"]
            label_ids = instance["label_ids"]

            # Update emissions matrix and label counts to contain the raw counts of each
            # token-label and label occurrence. Later, each value is divided by the label occurrence.
            for token, label in zip(token_ids, label_ids):

                # TODO: Smoothing needs to exist here.
                # E.g. ensure that the vocab has a fixed size so that
                # self.emission_matrix["@@UNKNOWN@@"][label] is non-zero.

                # Also need a function that does the UNK'ing stuff...
                self.emission_matrix[token][label] += 1

            # Update the ngram model. Each series of labels is prepended with
            # (order - 1) start tokens and appended with one end token.
            labels_ids_with_start_end = tuple(
                [self.start_token_id] * (self.order - 1) + label_ids + [self.end_token_id]
            )
            self._label_ngram_model.update(labels_ids_with_start_end)

            labels_with_start_end = [self.vocab.get_token_from_index(label, namespace=self.label_namespace)
                                     for label in labels_ids_with_start_end]

            train_text.append(labels_with_start_end)

        # TODO: `padded_everygram_pipeline` uses its own start and end tokens, figure out what to do about it
        train, nltk_vocab = padded_everygram_pipeline(self.order, train_text)

        self._lm_ngram_model.fit(train, nltk_vocab)

        self._construct_emission_matrix()
        self._construct_transition_matrix()

        self._is_trained = True

    def _construct_emission_matrix(self):

        # For each token, label pair, divide the label's frequency by
        # the frequency of the token to normalize.
        row_sums = self.emission_matrix.sum(axis=1)
        self.emission_matrix /= row_sums[:, np.newaxis]

    def _construct_transition_matrix(self):
        """
        Given the current state of the ngram model, construct a transition
        matrix such that each entry represents the maximum likelihood estimate
        of its index.

        Ex. For a trigram HMM, a transition matrix Q takes the form
                Q[W][U][V] = q(v | w, u)
            where q(v | w,  u) = c(w,  u, v) / c(w, u)
        """
        all_token_ids = list(range(0, self.vocab.get_vocab_size(namespace=self.label_namespace)))
        all_ngrams = itertools.product(*([all_token_ids] * self.order))
        for ngram in tqdm(all_ngrams):
            ngram_tokens = tuple(self.vocab.get_token_from_index(index, self.label_namespace) for index in ngram)
            word = ngram_tokens[-1]
            context = ngram_tokens[:-1]
            self.transition_matrix[ngram] = self._lm_ngram_model.score(word, context)

    def log_likelihood(self, input_tokens, labels):
        """
        Computes the log-likelihood of an input sequence and labels under this HMM.

        Assumes input tokens and labels are given as ids, and that there is 1:1 correspondence
        between given input tokens and labels.

        Parameters
        ----------
        input_tokens : List[int]
            The list of input tokens.
        labels : List[int]
            The list of labels.
        """

        # Conversion to log-space will cause division by zero warnings.
        np.seterr(divide="ignore")

        # pylint: disable=assignment-from-no-return
        emission_matrix = np.log2(self.emission_matrix)
        transition_matrix = np.log2(self.transition_matrix)

        if not self._is_trained:
            raise Exception(
                "Attempting to compute log likelihood with an untrained HMM."
            )

        if len(input_tokens) != len(labels):
            raise ValueError(
                "Received {} input tokens and {} labels but both should be the same length".format(
                    len(input_tokens), len(labels)
                )
            )

        # Emission likelihood in log space.
        emission_log_likelihood = np.zeros((1,))
        for input_token, label in zip(input_tokens, labels):
            emission_log_likelihood += emission_matrix[input_token][label]

        # Transition likelihood in log space.
        transition_log_likelihood = np.zeros((1,))

        # Augment labels with start and end tokens.
        labels = [self.start_token_id] * (self.order - 1) + labels + [self.end_token_id]

        ngrams = list(zip(*[labels[i:] for i in range(self.order)]))
        for ngram in ngrams:
            transition_log_likelihood += transition_matrix[ngram]

        return transition_log_likelihood + emission_log_likelihood

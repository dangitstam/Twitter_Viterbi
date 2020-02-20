import os
import pathlib

from viterbi.data.ark_tweet_nlp_conll_reader import read_ark_tweet_conll
from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import (
    DEFAULT_END_TOKEN,
    DEFAULT_LABEL_NAMESPACE,
    DEFAULT_START_TOKEN,
    DEFAULT_TOKEN_NAMESPACE,
    construct_vocab_from_dataset,
)
from viterbi.models.hidden_markov_model import HiddenMarkovModel

# pylint: disable=no-member
FIXTURES_ROOT = pathlib.Path(__file__).parent.absolute() / "fixtures"

# Construct a test dataset that rigs
# 1. An emission probability
# 2. A transition probability

TOKENS = ["token1", "token2", "token3", "token4", "token5"]

LABELS = ["label1", "label2", "label3", "label4", "label5"]


def construct_test_vocab_and_dataset():
    train_path = os.path.join(FIXTURES_ROOT, "hmm_test_dataset.txt")
    labels_file_path = os.path.join(FIXTURES_ROOT, "hmm_test_labels.txt")

    # Construct a vocabulary for both the tokens and label space from the dataset.
    vocab = construct_vocab_from_dataset(
        train_path,
        labels_file_path,
        read_ark_tweet_conll,
        label_namespace=DEFAULT_LABEL_NAMESPACE,
        token_namespace=DEFAULT_TOKEN_NAMESPACE,
        # The HMM prepends and appends start and end tokens before training. To do this, they first
        # have be added to the vocabulary so that they can be included when training the HMM.
        start_token=DEFAULT_START_TOKEN,
        end_token=DEFAULT_END_TOKEN,
    )

    # Construct a dataset reader and collect training instances.
    dataset_reader = DatasetReader(vocab, read_ark_tweet_conll)
    instances = dataset_reader.read(train_path)

    return vocab, instances


def verify_emission_matrix(vocab, instances, hmm):
    # All emission probabilities for token, label pairs in the dataset should be 1
    # except for token1 and token2 which were intentially swapped.
    token5_id = vocab.get_token_index("token5", namespace=DEFAULT_TOKEN_NAMESPACE)
    label5_id = vocab.get_token_index("label5", namespace=DEFAULT_LABEL_NAMESPACE)
    label6_id = vocab.get_token_index("label6", namespace=DEFAULT_LABEL_NAMESPACE)
    assert hmm.emission_matrix[token5_id][label5_id] == 0.5
    assert hmm.emission_matrix[token5_id][label6_id] == 0.5

    for i, token in enumerate(TOKENS):
        token_id = vocab.get_token_index(token, namespace=DEFAULT_TOKEN_NAMESPACE)
        for j, label in enumerate(LABELS):
            label_id = vocab.get_token_index(label, namespace=DEFAULT_LABEL_NAMESPACE)

            if token_id != token5_id:
                if i != j:
                    # For all tokens that aren't token5, the contrived case, they
                    # appear exactly twice with a unique token of the same number.
                    # E.g. token3 corresponds with label3.
                    #
                    # Therefore, emission probability is 1 when the number aligns, and 0 otherwise.
                    assert hmm.emission_matrix[token_id][label_id] == 0
                else:
                    assert hmm.emission_matrix[token_id][label_id] == 1
            else:
                if label_id != label5_id and label_id != label6_id:
                    # This covers the remaining portion of labels to test against the contrived
                    # tokens token1 and token2.
                    assert hmm.emission_matrix[token_id][label_id] == 0


def test_bigram_hidden_markov_model():
    """
    Tests order-2 HMM construction.
    """
    vocab, instances = construct_test_vocab_and_dataset()
    hmm = HiddenMarkovModel(
        vocab,
        order=2,
        label_namespace=DEFAULT_LABEL_NAMESPACE,
        token_namespace=DEFAULT_TOKEN_NAMESPACE,
        start_token=DEFAULT_START_TOKEN,
        end_token=DEFAULT_END_TOKEN,
    )
    hmm.train(instances)

    verify_emission_matrix(vocab, instances, hmm)

    # Verify bigram transition matrix construction.
    expected_bigrams = [
        ("label1", "label2"),
        ("label2", "label3"),
        ("label3", "label4"),
        ("label4", "label5"),
        # Results from the contrived case.
        ("label4", "label6"),
    ]

    expected_probability_one_bigrams = [
        ("label1", "label2"),
        ("label2", "label3"),
        ("label3", "label4"),
    ]

    for label_i in LABELS:
        label_i_id = vocab.get_token_index(label_i, namespace=DEFAULT_LABEL_NAMESPACE)
        for label_j in LABELS:
            label_j_id = vocab.get_token_index(
                label_j, namespace=DEFAULT_LABEL_NAMESPACE
            )
            if (label_i, label_j) not in expected_bigrams:
                assert hmm.transition_matrix[label_i_id][label_j_id] == 0
            elif (label_i, label_j) in expected_probability_one_bigrams:
                assert hmm.transition_matrix[label_i_id][label_j_id] == 1
            else:
                # Catches (label4, label5) and (label4, label6).
                assert hmm.transition_matrix[label_i_id][label_j_id] == 1 / 2


def test_trigram_hidden_markov_model():
    """
    Tests order-3 HMM construction.
    """
    vocab, instances = construct_test_vocab_and_dataset()
    hmm = HiddenMarkovModel(
        vocab,
        order=3,
        label_namespace=DEFAULT_LABEL_NAMESPACE,
        token_namespace=DEFAULT_TOKEN_NAMESPACE,
        start_token=DEFAULT_START_TOKEN,
        end_token=DEFAULT_END_TOKEN,
    )
    hmm.train(instances)

    verify_emission_matrix(vocab, instances, hmm)

    # Verify bigram transition matrix construction.
    expected_bigrams = [
        ("label1", "label2", "label3"),
        ("label2", "label3", "label4"),
        ("label3", "label4", "label5"),
        # Results from the contrived case.
        ("label3", "label4", "label6"),
    ]

    expected_probability_one_bigrams = [
        ("label1", "label2", "label3"),
        ("label2", "label3", "label4"),
    ]

    for label_i in LABELS:
        label_i_id = vocab.get_token_index(label_i, namespace=DEFAULT_LABEL_NAMESPACE)
        for label_j in LABELS:
            label_j_id = vocab.get_token_index(
                label_j, namespace=DEFAULT_LABEL_NAMESPACE
            )
            for label_k in LABELS:
                label_k_id = vocab.get_token_index(
                    label_k, namespace=DEFAULT_LABEL_NAMESPACE
                )
                if (label_i, label_j, label_k) not in expected_bigrams:
                    assert (
                        hmm.transition_matrix[label_i_id][label_j_id][label_k_id] == 0
                    )
                elif (label_i, label_j, label_k) in expected_probability_one_bigrams:
                    assert (
                        hmm.transition_matrix[label_i_id][label_j_id][label_k_id] == 1
                    )
                else:
                    # Catches (label3, label4, label5) and (label3, label4, label6).
                    assert (
                        hmm.transition_matrix[label_i_id][label_j_id][label_k_id]
                        == 1 / 2
                    )

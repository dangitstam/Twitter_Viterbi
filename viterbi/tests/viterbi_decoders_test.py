import argparse
import pathlib
from itertools import chain

import numpy as np
from tqdm import tqdm

from viterbi.data.ark_tweet_nlp_conll_reader import read_ark_tweet_conll
from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import (
    DEFAULT_END_TOKEN,
    DEFAULT_LABEL_NAMESPACE,
    DEFAULT_START_TOKEN,
    DEFAULT_TOKEN_NAMESPACE,
    construct_vocab_from_dataset,
)
from viterbi.environments import (
    ENVIRONMENTS,
    ark_tweet_conll_bigram_optimized,
    ark_tweet_conll_trigram,
    ark_tweet_conll_trigram_optimized,
)
from viterbi.models.hidden_markov_model import HiddenMarkovModel
from viterbi.models.viterbi_decoders import trigram_viterbi, viterbi
from viterbi.util import construct_model_from_environment

# pylint: disable=no-member
FIXTURES_ROOT = pathlib.Path(__file__).parent.absolute() / "fixtures"


def viterbi_log_likelihood_equals_hmm_log_likelihood_per_environment(
    train_path, label_set_path, environment, dev_path=None
):
    """
    The maximum likelihood derived by Viterbi should equal the log likelihood computed on the
    predicted sequence.
    """

    # Collect the dataset-specific parser. Behavior is undefined if this value is not specified
    # correctly (i.e. if the parser is incompatible with `train_path`).
    dataset_parser = environment["dataset_parser"]

    # Collect vocab parameters.
    label_namespace = environment["label_namespace"]
    start_token = environment["start_token"]
    end_token = environment["end_token"]

    model = construct_model_from_environment(train_path, label_set_path, environment)
    vocab = model["vocab"]
    hmm = model["hmm"]
    viterbi_decoder = model["viterbi_decoder"]

    # Evaluate model performance on the dev set.
    dataset_reader = DatasetReader(model["vocab"], dataset_parser)

    all_instances = dataset_reader.read(train_path)
    if dev_path:
        all_instances = chain(all_instances, dataset_reader.read(dev_path))

    for instance in all_instances:
        input_tokens = instance["token_ids"]
        output = viterbi_decoder(
            input_tokens,
            hmm.emission_matrix,
            hmm.transition_matrix,
            vocab.get_token_index(start_token, label_namespace),
            vocab.get_token_index(end_token, label_namespace),
        )
        log_likelihood = hmm.log_likelihood(instance["token_ids"], output["label_ids"])
        assert np.isclose(log_likelihood, output["log_likelihood"])


def test_viterbi_log_likelihood_equals_hmm_log_likelihood_bigram_optimized():
    label_set_path = FIXTURES_ROOT / "label_space.txt"
    train_path = FIXTURES_ROOT / "oct27.train"
    dev_path = FIXTURES_ROOT / "oct27.dev"
    viterbi_log_likelihood_equals_hmm_log_likelihood_per_environment(
        train_path, label_set_path, ark_tweet_conll_bigram_optimized, dev_path=dev_path
    )
    train_path = FIXTURES_ROOT / "daily547.conll"
    viterbi_log_likelihood_equals_hmm_log_likelihood_per_environment(
        train_path, label_set_path, ark_tweet_conll_bigram_optimized
    )


def test_viterbi_log_likelihood_equals_hmm_log_likelihood_trigram():
    label_set_path = FIXTURES_ROOT / "label_space.txt"
    train_path = FIXTURES_ROOT / "oct27.train"
    dev_path = FIXTURES_ROOT / "oct27.dev"
    environment = ark_tweet_conll_trigram

    dataset_parser = environment["dataset_parser"]
    label_namespace = environment["label_namespace"]
    start_token = environment["start_token"]
    end_token = environment["end_token"]

    model = construct_model_from_environment(train_path, label_set_path, environment)
    vocab = model["vocab"]
    hmm = model["hmm"]

    dataset_reader = DatasetReader(model["vocab"], dataset_parser)

    all_instances = dataset_reader.read(train_path)
    if dev_path:
        all_instances = chain(all_instances, dataset_reader.read(dev_path))

    for instance in all_instances:
        input_tokens = instance["token_ids"]
        trigram_output = viterbi(
            input_tokens,
            hmm.emission_matrix,
            hmm.transition_matrix,
            vocab.get_token_index(start_token, label_namespace),
            vocab.get_token_index(end_token, label_namespace),
        )
        log_likelihood = hmm.log_likelihood(input_tokens, trigram_output["label_ids"])
        assert np.isclose(log_likelihood, trigram_output["log_likelihood"])


def test_viterbi_log_likelihood_equals_hmm_log_likelihood_trigram_optimized():
    label_set_path = FIXTURES_ROOT / "label_space.txt"
    train_path = FIXTURES_ROOT / "oct27.train"
    dev_path = FIXTURES_ROOT / "oct27.dev"
    viterbi_log_likelihood_equals_hmm_log_likelihood_per_environment(
        train_path, label_set_path, ark_tweet_conll_trigram_optimized, dev_path=dev_path
    )
    train_path = FIXTURES_ROOT / "daily547.conll"
    viterbi_log_likelihood_equals_hmm_log_likelihood_per_environment(
        train_path, label_set_path, ark_tweet_conll_trigram_optimized
    )


def test_optimized_trigram_equals_trigram():
    label_set_path = FIXTURES_ROOT / "label_space.txt"
    train_path = FIXTURES_ROOT / "oct27.train"
    dev_path = FIXTURES_ROOT / "oct27.dev"
    environment = ark_tweet_conll_trigram

    # Collect the dataset-specific parser. Behavior is undefined if this value is not specified
    # correctly (i.e. if the parser is incompatible with `train_path`).
    dataset_parser = environment["dataset_parser"]

    # Collect vocab parameters.
    label_namespace = environment["label_namespace"]
    start_token = environment["start_token"]
    end_token = environment["end_token"]

    model = construct_model_from_environment(train_path, label_set_path, environment)
    vocab = model["vocab"]
    hmm = model["hmm"]

    # Evaluate model performance on the dev set.
    dataset_reader = DatasetReader(model["vocab"], dataset_parser)

    all_instances = dataset_reader.read(train_path)
    if dev_path:
        all_instances = chain(all_instances, dataset_reader.read(dev_path))

    for instance in all_instances:
        input_tokens = instance["token_ids"]
        trigram_output = viterbi(
            input_tokens,
            hmm.emission_matrix,
            hmm.transition_matrix,
            vocab.get_token_index(start_token, label_namespace),
            vocab.get_token_index(end_token, label_namespace),
        )

        optimized_trigram_output = viterbi(
            input_tokens,
            hmm.emission_matrix,
            hmm.transition_matrix,
            vocab.get_token_index(start_token, label_namespace),
            vocab.get_token_index(end_token, label_namespace),
        )

        assert np.isclose(
            trigram_output["log_likelihood"], optimized_trigram_output["log_likelihood"]
        )
        assert trigram_output["label_ids"] == optimized_trigram_output["label_ids"]

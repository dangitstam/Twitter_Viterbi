import pathlib
from itertools import chain

import numpy as np

from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import construct_vocab_from_dataset
from viterbi.environments import (
    ark_tweet_conll_bigram_optimized,
    ark_tweet_conll_trigram,
    ark_tweet_conll_trigram_optimized,
)
from viterbi.models.hidden_markov_model import HiddenMarkovModel
from viterbi.models.viterbi_decoders import trigram_viterbi, viterbi

# pylint: disable=no-member
FIXTURES_ROOT = pathlib.Path(__file__).parent.absolute() / "fixtures"


def construct_model_from_environment(train_path, label_set_path, environment):
    """
    A helper function for quickly standing up a vocabulary and hidden markov model.

    Returns a dictionary containing the vocab, hmm, and chosen viterbi decoder.
    """

    # Collect the dataset-specific parser. Behavior is undefined if this value is not specified
    # correctly (i.e. if the parser is incompatible with `train_path`).
    dataset_parser = environment["dataset_parser"]

    # Collect vocab parameters.
    token_namespace = environment["token_namespace"]
    label_namespace = environment["label_namespace"]
    start_token = environment["start_token"]
    end_token = environment["end_token"]
    max_vocab_size = environment["max_vocab_size"]
    min_count = environment["min_count"]

    # Collect HMM and Viterbi parameters.
    order = environment["order"]
    viterbi_decoder = environment["viterbi_decoder"]

    # Construct a vocabulary for both the tokens and label space from the dataset.
    vocab = construct_vocab_from_dataset(
        train_path,
        dataset_parser,
        label_set_path=label_set_path,
        token_namespace=token_namespace,
        label_namespace=label_namespace,
        max_vocab_size=max_vocab_size,
        min_count=min_count,
        # The HMM prepends and appends start and end tokens before training. To do this, they first
        # have be added to the vocabulary so that they can be included when training the HMM.
        start_token=start_token,
        end_token=end_token,
    )

    # Construct a dataset reader and collect training instances.
    dataset_reader = DatasetReader(vocab, dataset_parser)
    instances = dataset_reader.read(train_path)

    # Train a hidden markov model to learn transition and emission probabilities.
    hmm = HiddenMarkovModel(
        vocab,
        order=order,
        token_namespace=token_namespace,
        label_namespace=label_namespace,
    )
    hmm.train(instances)

    output = {
        "hmm": hmm,
        "viterbi_decoder": viterbi_decoder,
        "vocab": vocab,
        "dataset_reader": dataset_reader,
    }

    return output


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

    # Too verbose to check all instances, 50 should be plenty.
    for i, instance in enumerate(all_instances):
        input_tokens = instance["token_ids"]
        trigram_output = trigram_viterbi(
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

        if i > 50:
            break

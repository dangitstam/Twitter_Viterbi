"""
TODO: Arg parser, demo mode, accept input produce output as files, unit tests.

To unit test viterbi
* Assert the expected value is the highest in both sequence probability and emission before testing.

"""

import argparse

from viterbi.data.ark_tweet_nlp_conll_reader import read_ark_tweet_conll
from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import (
    construct_vocab_from_dataset,
    DEFAULT_TOKEN_NAMESPACE,
    DEFAULT_LABEL_NAMESPACE,
)
from viterbi.models.hidden_markov_model import HiddenMarkovModel
from viterbi.models.viterbi import trigram_viterbi, viterbi


# TODO: Dict mapping readers to a name, put in data dir
# e.g.
dataset_parsers = {"ark-tweet-conll": read_ark_tweet_conll}

# TODO: token and label namespaces, if not provided, should be constants somewhere else.


def main():
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument(
    #     "--train-path", type=str, required=True, help="Path to the training set."
    # )
    # parser.add_argument(
    #     "--dev-path", type=str, required=True, help="Path to the dev set."
    # )
    # parser.add_argument(
    #     "--order", type=int, default=3, help="The degree of n for the ngram model."
    # )
    # parser.add_argument(
    #     "--serialization-dir",
    #     "-s",
    #     type=str,
    #     required=True,
    #     help="Path to store the preprocessed output.",
    # )
    # parser.add_argument(
    #     "--vocab-size",
    #     type=int,
    #     required=False,
    #     default=10000,
    #     help="Path to store the preprocessed corpus vocabulary (output file name).",
    # )
    # args = parser.parse_args()

    # Construct a vocabulary for both the tokens and label space from the dataset.
    token_namespace = "tokens"
    label_namespace = "labels"
    vocab = construct_vocab_from_dataset(
        "/Users/dangitstam/Datasets/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.conll",
        "/Users/dangitstam/Git/Twitter_Viterbi/label_space.txt",
        read_ark_tweet_conll,
        token_namespace=token_namespace,
        label_namespace=label_namespace,
    )

    # TODO: Store these as constants.
    # The HMM will prepends and appends start and end tokens. To do this, they
    # first must be added to the vocabulary and then specified when building the HMM.
    start_token = "@@START@@"
    end_token = "@@END@@"
    vocab.add_token_to_namespace(start_token, label_namespace)
    vocab.add_token_to_namespace(end_token, label_namespace)

    # Construct a dataset reader.
    dataset_reader = DatasetReader(vocab, read_ark_tweet_conll)
    instances = dataset_reader.read(
        "/Users/dangitstam/Datasets/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.conll"
    )

    # Train a hidden markov model (learn transition and emission probabilities).
    hmm = HiddenMarkovModel(
        vocab, order=3, token_namespace=token_namespace, label_namespace=label_namespace
    )

    hmm.train(instances)

    instances = dataset_reader.read(
        "/Users/dangitstam/Datasets/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.conll"
    )

    for i in range(100):
        first = next(instances)

        input_tokens = first["token_ids"]

        output = viterbi(
            input_tokens,
            hmm.emission_matrix,
            hmm.transition_matrix,
            vocab.get_token_index(start_token, label_namespace),
            vocab.get_token_index(end_token, label_namespace),
        )

        prediction_labels = list(
            map(
                lambda x: vocab.get_token_from_index(x, label_namespace),
                output["label_ids"],
            )
        )

        log_likelihood = hmm.log_likelihood(first["token_ids"], output["label_ids"])

        print(log_likelihood - output["log_likelihood"])


if __name__ == "__main__":
    main()

"""
TODO: Arg parser, demo mode, accept input produce output as files, unit tests.

To unit test viterbi
* Assert the expected value is the highest in both sequence probability and emission before testing.

"""

import argparse
import logging
import pprint

from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import construct_vocab_from_dataset
from viterbi.data.util import twitter_unk
from viterbi.environments import ENVIRONMENTS
from viterbi.models.model import Model


def main():
    logging.basicConfig(
        filename="training.log",
        filemode="r",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path", type=str, required=True, help="Path to the training set."
    )
    parser.add_argument(
        "--dev-path", type=str, required=True, help="Path to the dev set."
    )
    parser.add_argument(
        "--label-set-path",
        type=str,
        required=True,
        help="Path to file containing the state space.",
    )
    parser.add_argument(
        "--environment", type=str, required=True, help="The model configuration."
    )
    parser.add_argument(
        "--serialization-dir", type=str, required=True, help="Path to store the output."
    )
    args = parser.parse_args()

    train_path = args.train_path
    dev_path = args.dev_path
    label_set_path = args.label_set_path

    if args.environment not in ENVIRONMENTS:
        raise ValueError("Unsupported environment: {}".format(args.environment))

    environment = ENVIRONMENTS.get(args.environment)

    # Collect the dataset-specific parser. Behavior is undefined if this value is not specified
    # correctly (i.e. if the parser is incompatible with `train_path`).
    dataset_parser = environment.get("dataset_parser")

    # Collect vocab parameters.
    token_namespace = environment.get("token_namespace")
    label_namespace = environment.get("label_namespace")
    start_token = environment.get("start_token")
    end_token = environment.get("end_token")
    max_vocab_size = environment.get("max_vocab_size")
    min_count = environment.get("min_count")
    lowercase_tokens = environment.get("lowercase_tokens")
    special_unknown_token_fn = environment.get("special_unknown_token_fn")

    # Collect HMM and Viterbi parameters.
    order = environment.get("order")
    viterbi_decoder = environment.get("viterbi_decoder")

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
        lowercase_tokens=lowercase_tokens,
    )

    # Construct a dataset reader and collect training instances.
    def token_preprocessing_fn(tokens):
        if lowercase_tokens:
            tokens = map(tokens.lower(), tokens)
        if special_unknown_token_fn:
            tokens = map(twitter_unk, tokens)
        return list(tokens)

    logging.info("reading dataset")
    dataset_reader = DatasetReader(
        vocab, dataset_parser, token_preprocessing_fn=token_preprocessing_fn
    )
    instances = dataset_reader.read(train_path)

    logging.info("training model")
    model = Model(
        order,
        viterbi_decoder,
        vocab,
        start_token,
        end_token,
        token_namespace,
        label_namespace,
    )
    model.train_model(instances)

    logging.info("model validation")
    dev_instances = dataset_reader.read(dev_path)
    dev_results = model.evaluate(dev_instances)
    pprint.pprint(dev_results, width=4)


if __name__ == "__main__":
    main()

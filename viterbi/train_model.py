"""
TODO: Arg parser, demo mode, accept input produce output as files, unit tests.

To unit test viterbi
* Assert the expected value is the highest in both sequence probability and emission before testing.

"""

import argparse

from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import construct_vocab_from_dataset
from viterbi.models.hidden_markov_model import HiddenMarkovModel

from viterbi.environments import ENVIRONMENTS


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train-path", type=str, required=True, help="Path to the training set."
    )
    parser.add_argument(
        "--dev-path", type=str, required=True, help="Path to the dev set."
    )

    # TODO: Make this optional, infer the state-space from the training data.
    # When a new label is seen during training, throw an exception (this should be true already anyway).
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

    environment = ENVIRONMENTS[args.environment]

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
        label_set_path,
        dataset_parser,
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

    # TODO: In this script,
    # TODO: Produce a dict containing a trained model and its vocab, and then output that dict as a tarball to the serialization dir.
    # TODO: Does it make sense to make a "model" that takes an HMM, vocab, and viterbi decoder to define a clean forward function?
    output = {"hmm": hmm, "viterbi_decoder": viterbi_decoder, "vocab": vocab}

    # Evaluate model performance on the dev set.
    dev_instances = dataset_reader.read(dev_path)
    correctly_tagged_words = 0
    total_words = 0
    for instance in dev_instances:
        input_tokens = instance["token_ids"]

        output = viterbi_decoder(
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

        labels = instance["labels"]

        correctly_labeled = [
            int(prediction == label)
            for prediction, label in zip(prediction_labels, labels)
        ]
        correctly_tagged_words += sum(correctly_labeled)
        total_words += len(labels)

        log_likelihood = hmm.log_likelihood(instance["token_ids"], output["label_ids"])
        import numpy as np

        assert np.isclose(log_likelihood, output["log_likelihood"])

    print(correctly_tagged_words / total_words)


if __name__ == "__main__":
    main()

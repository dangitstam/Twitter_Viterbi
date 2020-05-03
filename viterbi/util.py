from viterbi.data.dataset_reader import DatasetReader
from viterbi.data.util import construct_vocab_from_dataset
from viterbi.models.hidden_markov_model import HiddenMarkovModel


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

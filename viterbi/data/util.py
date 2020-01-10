import re
from collections import Counter

from allennlp.data.vocabulary import Vocabulary


# Default namespaces for tokens and labels.
DEFAULT_TOKEN_NAMESPACE = "tokens"
DEFAULT_LABEL_NAMESPACE = "labels"


def construct_vocab_from_dataset(
    train_file_path,
    labels_file_path,
    reader,
    min_count=None,
    max_vocab_size=None,
    token_namespace="tokens",
    label_namespace="labels",
):
    """
    Constructs an AllenNLP vocabulary from the given dataset with two separate
    namespaces: tokens and labels.

    Paramaters
    ----------
    file_path : The file path of the dataset.
    reader : A first class function that can parse the dataset in `file_path`
             and return an generator of instances over the dataset. Assumes
             each instance is a Tuple[List[Str]], the first being the list
             of tokens and the second being a list of labels
    min_count : The minimum number of instances a token has to occur to be
                included.
    max_vocab_size : The maximum vocab size.
    """

    instances = reader(train_file_path)

    token_counts = Counter()
    for tokens, _ in instances:
        token_counts.update(tokens)

    labels_file_text = open(labels_file_path, "r").read()
    labels = [l for l in re.split("\n+", labels_file_text) if l]

    counter = {token_namespace: token_counts}

    vocab = Vocabulary(
        counter=counter,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
        non_padded_namespaces=[label_namespace],
    )

    for label in labels:
        vocab.add_token_to_namespace(label, label_namespace)

    return vocab

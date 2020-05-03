import re
from collections import Counter

from allennlp.data.vocabulary import Vocabulary

# Default namespaces for tokens and labels.
DEFAULT_TOKEN_NAMESPACE = "tokens"
DEFAULT_LABEL_NAMESPACE = "labels"

# TODO: These and env-defined defaults are not aligned.
DEFAULT_START_TOKEN = "<s>"
DEFAULT_END_TOKEN = "</s>"


def construct_vocab_from_dataset(
    train_file_path,
    reader,
    label_set_path=None,
    min_count=None,
    max_vocab_size=None,
    token_namespace=DEFAULT_TOKEN_NAMESPACE,
    label_namespace=DEFAULT_LABEL_NAMESPACE,
    start_token=None,
    end_token=None,
):
    """
    Constructs an AllenNLP vocabulary from the given dataset with two separate
    namespaces: tokens and labels.

    Parameters
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
    label_set = set()
    for tokens, labels in instances:
        token_counts.update(tokens)
        label_set.update(labels)

    label_set = list(label_set)

    # TODO: For some reason, changing the label set from a list to a set
    # results in non-deterministic behavior in validation accuracy.
    #
    # There is no randomness involved, yet changing labels from a list to
    # a set results in validation hovering between 0.65 and 0.67 on the
    # twitter dataset.
    if label_set_path:
        labels_file_text = open(label_set_path, "r").read()
        label_set = set([l for l in re.split("\n+", labels_file_text) if l])

    counter = {token_namespace: token_counts}

    # TODO: When the start and end tokens are added last, there is a bug.
    label_set.update([start_token, end_token])

    # TODO: min_count should be a dictionary.
    vocab = Vocabulary(
        counter=counter,
        min_count=min_count,
        max_vocab_size={token_namespace: max_vocab_size},
        tokens_to_add={
            label_namespace: label_set
        }
    )

    return vocab

import re
from collections import Counter

from allennlp.data.vocabulary import Vocabulary

# Default namespaces for tokens and labels.
DEFAULT_TOKEN_NAMESPACE = "tokens"
DEFAULT_LABEL_NAMESPACE = "labels"

DEFAULT_START_TOKEN = "@@START@@"
DEFAULT_END_TOKEN = "@@END@@"


# Unknown symbols and UNK'ing specifically made for Twitter data.
HASHTAG = "@@HASHTAG@@"
MENTION = "@@MENTION@@"
NUMERIC = "@@NUMERIC@@"
URL = "@@URL@@"
PUNCT = "@@PUNCT@@"
SPECIALIZED_UNKNOWNS = {HASHTAG, MENTION, NUMERIC, URL, PUNCT}


def twitter_unk(xi, vocab: Vocabulary, token_namespace):
    """
    Uses regular expressions to look for tell-tale signs
    that prove beyond doubt that a word takes a particular
    form.

    If the word does not match a heuristic, defaults
    to <UNK> if it is an unknown word.
    """

    if re.match(r"^@.*", xi):
        # Twitter handles are restricted to alphanumerics
        # and underscores.
        xi = MENTION
    elif re.match(r'^[$\'",.!?]+$', xi):
        # Matches against arbitrary lengths of punctuation.
        # Prioritized before HASHTAG to catch punctuation-only words.
        # Colon left out to prevent accidentally matching against emoticons.
        xi = PUNCT
    elif re.match(r"^#[^#]+$", xi):
        # Any non-punct word begining with '#' in a tweet is a hashtag.
        xi = HASHTAG
    elif re.match(r"^http[s]?://.+", xi) or re.match(
        r"^[a-zA-Z0-9_\.]+@[a-zA-Z0-9_\.]+", xi
    ):
        # Match URLs and Emails
        xi = URL
    elif xi.isdigit() or re.match(r"^[0-9]+[:.-x][0-9]+", xi):
        # Matches numbers and times (ex. 2010, 9:30)
        xi = NUMERIC

    return xi


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
    lowercase_tokens=False,
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
        if lowercase_tokens:
            tokens = map(lambda token: token.lower(), tokens)

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
            token_namespace: SPECIALIZED_UNKNOWNS,
            label_namespace: label_set,
        },
    )

    return vocab

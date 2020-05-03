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
EMAIL = "@@EMAIL@@"
SPECIALIZED_UNKNOWNS = {HASHTAG, MENTION, NUMERIC, URL, PUNCT, EMAIL}


def twitter_unk(token: str) -> str:
    """
    When given a token, returns either the token itself, or one of
    "@@HASHTAG@@", "@@MENTION@@", "@@NUMERIC@@", "@@URL@@", "@@PUNCT@@"
    depending on the form of the token.

    Parameters
    ----------
    token : The token to match against.
    """

    if re.match(r"^@[a-zA-Z0-9_]+", token):
        # Twitter handles are restricted to alphanumerics
        # and underscores.
        token = MENTION
    elif re.match(r'^[$\'",.!?]+$', token):
        # Matches against arbitrary lengths of punctuation.
        # Prioritized before HASHTAG to catch punctuation-only words.
        # Colon left out to prevent accidentally matching against emoticons.
        token = PUNCT
    elif re.match(r"^#[^#]+$", token):
        # Any non-punctuation word beginning with '#' in a tweet is a hashtag.
        token = HASHTAG
    elif re.match(r"^http[s]?://.+", token) or re.match(
        r"^[a-zA-Z0-9_.]+@[a-zA-Z0-9_.]+", token
    ):
        # Match URLs
        token = URL
    elif re.match(r"^[a-zA-Z0-9_.]+@[a-zA-Z0-9_.]+", token):
        # Match Emails
        token = EMAIL
    elif token.isdigit() or re.match(r"^[0-9]+[:.-x][0-9]+", token):
        # Matches numbers and times (ex. 2010, 9:30)
        token = NUMERIC

    return token


def construct_vocab_from_dataset(
    train_file_path: str,
    reader,
    label_set_path: str = None,
    min_count=None,
    max_vocab_size: int = None,
    token_namespace: str = DEFAULT_TOKEN_NAMESPACE,
    label_namespace: str = DEFAULT_LABEL_NAMESPACE,
    start_token: str = None,
    end_token: str = None,
    lowercase_tokens: bool = False,
) -> Vocabulary:
    """
    Constructs an AllenNLP vocabulary from the given dataset with two separate
    namespaces: tokens and labels.

    Parameters
    ----------
    train_file_path : The file path of the dataset.
    reader : A first class function that can parse the dataset in `file_path`
             and return an generator of instances over the dataset. Assumes
             each instance is a Tuple[List[Str]], the first being the list
             of tokens and the second being a list of labels
    label_set_path: If given, creates the label set from this file. Expected
                    to be a text file where each line is a single label.
    min_count : The minimum number of instances a token has to occur to be
                included.
    max_vocab_size : The maximum vocab size.
    token_namespace: The namespace of the input tokens.
    label_namespace: The namespace of the input labels.
    start_token: The start token, when provided, is added to the vocabulary.
    end_token: The end token, when provided, is added to the vocabulary.
    lowercase_tokens: When true, results in a vocabulary of only lowercase tokens.
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
    # for trigram viterbi results in non-deterministic behavior in validation accuracy.
    #
    # There is no randomness involved, yet changing labels from a list to
    # a set results in validation hovering between 0.65 and 0.67 on the
    # twitter dataset.
    if label_set_path:
        labels_file_text = open(label_set_path, "r").read()
        label_set = set([label for label in re.split("\n+", labels_file_text) if label])

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

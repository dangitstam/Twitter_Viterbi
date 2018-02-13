
import re

""" Unknown symbols and UNK'ing specifically made for Twitter data. """

START = "<START>"
STOP = "<STOP>"    
UNKNOWN = "<UNK>"
HASHTAG = "<HASHTAG>"
MENTION = "<MENTION>"
NUMERIC = "<NUMERIC>"
URL = "<URL>"
PUNCT = "<PUNCT>"

UNKS = set([START, STOP, UNKNOWN, HASHTAG, MENTION, NUMERIC, URL, PUNCT])


def twitter_unk(xi, vocab):
    # Uses regular expressions to look for tell-tale signs
    # that prove beyond doubt that a word takes a particular
    # form.
    #
    # If the word does not match a heuristic, defaults
    # to <UNK> if it is an unknown word.
    if xi not in vocab:
        if re.match(r'^@[a-zA-Z0-9_]+', xi):
            # Twitter handles are restricted to alphanumerics
            # and underscores.
            xi = MENTION
        elif re.match(r'^[$\'",.!?]+$', xi):
            # Matches against arbitrary lengths of punctuation.
            # Priortized before HASHTAG to catch punct-only words.
            # Colon left out to prevent accidentally matching against
            # emoticons.
            xi = PUNCT
        elif re.match(r'^#[^#]+$', xi):
            # Any non-punct word begining with '#' in a tweet is a hashtag.
            xi = HASHTAG
        elif re.match(r'^http[s]?://.+', xi) or re.match(r'^[a-zA-Z0-9_\.]+@[a-zA-Z0-9_\.]+', xi):
            # Match URLs and Emails
            xi = URL
        elif xi.isdigit() or re.match(r'^[0-9]+[:.-x][0-9]+', xi):
            # Matches numbers and times (ex. 2010, 9:30)
            xi = NUMERIC
        else:
            xi = UNKNOWN

    return xi


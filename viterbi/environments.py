from viterbi.data.ark_tweet_nlp_conll_reader import read_ark_tweet_conll
from viterbi.data.util import (
    DEFAULT_END_TOKEN,
    DEFAULT_LABEL_NAMESPACE,
    DEFAULT_START_TOKEN,
    DEFAULT_TOKEN_NAMESPACE,
)
from viterbi.data.util import twitter_unk
from viterbi.models.viterbi_decoders import trigram_viterbi, viterbi


# Yields ~71.2% accuracy on the CMU twitter dataset.
ark_tweet_conll_trigram = {
    "dataset_parser": read_ark_tweet_conll,
    "token_namespace": DEFAULT_TOKEN_NAMESPACE,
    "label_namespace": DEFAULT_LABEL_NAMESPACE,
    "start_token": DEFAULT_START_TOKEN,
    "end_token": DEFAULT_END_TOKEN,
    "viterbi_decoder": trigram_viterbi,
    "order": 3,
    "max_vocab_size": 5000,
    "min_count": None,
    "lowercase_tokens": False,
    "special_unknown_token_fn": twitter_unk,
}

ark_tweet_conll_bigram_optimized = {
    "dataset_parser": read_ark_tweet_conll,
    "token_namespace": DEFAULT_TOKEN_NAMESPACE,
    "label_namespace": DEFAULT_LABEL_NAMESPACE,
    "start_token": DEFAULT_START_TOKEN,
    "end_token": DEFAULT_END_TOKEN,
    "viterbi_decoder": viterbi,
    "order": 2,
    "max_vocab_size": 250,
    "min_count": None,
    "lowercase_tokens": False,
    "special_unknown_token_fn": twitter_unk,
}

# Yields ~70% accuracy on the CMU twitter dataset.
ark_tweet_conll_trigram_optimized = {
    "dataset_parser": read_ark_tweet_conll,
    "token_namespace": DEFAULT_TOKEN_NAMESPACE,
    "label_namespace": DEFAULT_LABEL_NAMESPACE,
    "start_token": DEFAULT_START_TOKEN,
    "end_token": DEFAULT_END_TOKEN,
    "viterbi_decoder": viterbi,
    "order": 3,
    "max_vocab_size": 225,
    "min_count": None,
    "lowercase_tokens": True,
    "special_unknown_token_fn": twitter_unk,
}

ENVIRONMENTS = {
    "ark_tweet_conll_trigram": ark_tweet_conll_trigram,
    "ark_tweet_conll_trigram_optimized": ark_tweet_conll_trigram_optimized,
    "ark_tweet_conll_bigram_optimized": ark_tweet_conll_bigram_optimized,
}

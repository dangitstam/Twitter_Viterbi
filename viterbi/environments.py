from viterbi.data.ark_tweet_nlp_conll_reader import read_ark_tweet_conll
from viterbi.data.util import (
    DEFAULT_END_TOKEN,
    DEFAULT_LABEL_NAMESPACE,
    DEFAULT_START_TOKEN,
    DEFAULT_TOKEN_NAMESPACE,
)
from viterbi.models.viterbi_decoders import trigram_viterbi, viterbi

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
}

ark_tweet_conll_bigram_optimized = {
    "dataset_parser": read_ark_tweet_conll,
    "token_namespace": DEFAULT_TOKEN_NAMESPACE,
    "label_namespace": DEFAULT_LABEL_NAMESPACE,
    "start_token": DEFAULT_START_TOKEN,
    "end_token": DEFAULT_END_TOKEN,
    "viterbi_decoder": viterbi,
    "order": 2,
    "max_vocab_size": 5000,
    "min_count": None,
}

ark_tweet_conll_trigram_optimized = {
    "dataset_parser": read_ark_tweet_conll,
    "token_namespace": DEFAULT_TOKEN_NAMESPACE,
    "label_namespace": DEFAULT_LABEL_NAMESPACE,
    "start_token": DEFAULT_START_TOKEN,
    "end_token": DEFAULT_END_TOKEN,
    "viterbi_decoder": viterbi,
    "order": 3,
    "max_vocab_size": 5100,
    "min_count": None,
}

ENVIRONMENTS = {
    "ark_tweet_conll_trigram": ark_tweet_conll_trigram,
    "ark_tweet_conll_trigram_optimized": ark_tweet_conll_trigram_optimized,
}

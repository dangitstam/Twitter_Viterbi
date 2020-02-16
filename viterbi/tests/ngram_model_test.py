import numpy as np

from viterbi.models.ngram_model import NGramModel


def test_unigram_ngram_model_update():
    """
    Unigram models should have empty contexts and a unigram for each token in the input.
    """

    ngram_model = NGramModel(1)
    assert len(ngram_model._ngram_frequencies) == 0
    assert len(ngram_model._context_frequencies) == 0

    input_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    expected_unigrams = [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
        ("e",),
        ("f",),
        ("g",),
        ("h",),
        ("i",),
    ]

    ngram_model.update(input_tokens)
    assert len(ngram_model._ngram_frequencies) == len(expected_unigrams)
    assert len(ngram_model._context_frequencies) == 0

    for unigram in expected_unigrams:
        assert unigram in ngram_model._ngram_frequencies


def test_bigram_ngram_model_update():
    """
    Bigram models should have each unigram in the context dictionary, and
    should account for each possible trigram..
    """

    ngram_model = NGramModel(2)
    assert len(ngram_model._ngram_frequencies) == 0
    assert len(ngram_model._context_frequencies) == 0

    input_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    expected_bigrams = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
        ("g", "h"),
        ("h", "i"),
    ]
    expected_unigrams = [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
        ("e",),
        ("f",),
        ("g",),
        ("h",),
        ("i",),
    ]

    ngram_model.update(input_tokens)
    assert len(ngram_model._ngram_frequencies) == len(expected_bigrams)
    assert len(ngram_model._context_frequencies) == len(expected_unigrams)

    for bigram in expected_bigrams:
        assert bigram in ngram_model._ngram_frequencies

    for unigram in expected_unigrams:
        assert unigram in ngram_model._context_frequencies


def test_trigram_ngram_model_update():
    """
    Trigram models should have each possible bigram in the context dictionary, and
    should account for each possible trigram. 
    """

    ngram_model = NGramModel(3)
    assert len(ngram_model._ngram_frequencies) == 0
    assert len(ngram_model._context_frequencies) == 0

    input_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    expected_trigrams = [
        ("a", "b", "c"),
        ("b", "c", "d"),
        ("c", "d", "e"),
        ("d", "e", "f"),
        ("e", "f", "g"),
        ("f", "g", "h"),
        ("g", "h", "i"),
    ]

    expected_bigrams = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
        ("g", "h"),
        ("h", "i"),
    ]

    ngram_model.update(input_tokens)
    assert len(ngram_model._ngram_frequencies) == len(expected_trigrams)
    assert len(ngram_model._context_frequencies) == len(expected_bigrams)

    for trigram in expected_trigrams:
        assert trigram in ngram_model._ngram_frequencies

    for bigram in expected_bigrams:
        assert bigram in ngram_model._context_frequencies


def test_unigram_ngram_model_mle():
    ngram_model = NGramModel(1)
    assert len(ngram_model._ngram_frequencies) == 0
    assert len(ngram_model._context_frequencies) == 0

    input_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "a", "b", "c"]
    expected_unigrams = [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
        ("e",),
        ("f",),
        ("g",),
        ("h",),
        ("i",),
    ]

    ngram_model.update(input_tokens)
    duplicates = set([("a",), ("b",), ("c",)])
    for unigram in expected_unigrams:
        if unigram in duplicates:
            assert np.isclose(
                ngram_model.maximum_likelihood_estimate(unigram), 2 / len(input_tokens)
            )
        else:
            assert np.isclose(
                ngram_model.maximum_likelihood_estimate(unigram), 1 / len(input_tokens)
            )


def test_bigram_ngram_model_mle():
    """
    Tests that maximum likelihood estimates are computed correctly by introducing a bigram
    ("a", "b") whose context ("a",) also appears in more places.
    """

    ngram_model = NGramModel(2)
    assert len(ngram_model._ngram_frequencies) == 0
    assert len(ngram_model._context_frequencies) == 0

    input_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "a"]
    expected_bigrams = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
        ("g", "h"),
        ("h", "i"),
        ("i", "a"),  # Introduced through duplicate 'a'.
    ]

    ngram_model.update(input_tokens)
    for bigram in expected_bigrams:
        if bigram == ("a", "b"):
            # MLE is the count of the ngram divided by the count of the frequency of
            # its context. In `input_tokens`, "a" appears twice and  "a", "b" appears once.
            assert np.isclose(ngram_model.maximum_likelihood_estimate(bigram), 0.5)
        else:
            assert np.isclose(ngram_model.maximum_likelihood_estimate(bigram), 1)


def test_trigram_ngram_model_mle():
    """
    Tests that maximum likelihood estimates are computed correctly by introducing a bigram
    ("a", "b", "c") whose context ("a", "b") also appears in more places.
    """
    ngram_model = NGramModel(3)
    assert len(ngram_model._ngram_frequencies) == 0
    assert len(ngram_model._context_frequencies) == 0

    input_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "a", "b", "x"]
    expected_trigrams = [
        ("a", "b", "c"),
        ("b", "c", "d"),
        ("c", "d", "e"),
        ("d", "e", "f"),
        ("e", "f", "g"),
        ("f", "g", "h"),
        ("g", "h", "i"),
        # Introduced through duplicate "a", "b", and "c".
        ("h", "i", "a"),
        ("i", "a", "b"),
    ]

    ngram_model.update(input_tokens)
    for trigram in expected_trigrams:
        if trigram == ("a", "b", "c"):
            # ("a", "b", "x") appears once, but "a", "b" appears twice.
            assert np.isclose(ngram_model.maximum_likelihood_estimate(trigram), 0.5)
        else:
            assert np.isclose(ngram_model.maximum_likelihood_estimate(trigram), 1)

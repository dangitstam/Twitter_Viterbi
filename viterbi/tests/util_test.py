from viterbi.data.util import HASHTAG, MENTION, NUMERIC, URL, PUNCT, twitter_unk

tests = {
    # Twitter mentions
    "@e_one": MENTION,  # Problem during development.
    "@dangitstam": MENTION,
    # URLs and Emails.
    "email@example.com": URL,
    "firstname.lastname@example.com": URL,
    "email@subdomain.example.com": URL,
    "firstname+lastname@example.com": URL,
    "email@123.123.123.123": URL,
    "email@[123.123.123.123]": URL,
    '"email"@example.com': URL,
    "1234567890@example.com": URL,
    "email@example-one.com": URL,
    "_______@example.com": URL,
    "email@example.name": URL,
    "email@example.museum": URL,
    "email@example.co.jp": URL,
    "firstname-lastname@example.com": URL,
    # Hashtags
    "#PutACanOnIt": HASHTAG,
    "#ShareaCoke": HASHTAG,
    "#TweetFromTheSeat": HASHTAG,
    "#OreoHorrorStories": HASHTAG,
    "#WantAnR8": HASHTAG,
    "#NationalFriedChickenDay": HASHTAG,
    "#CollegeIn5Words": HASHTAG,
    # Punctuation
    ",": PUNCT,
    ".": PUNCT,
    ":": PUNCT,
    "!": PUNCT,
    "?": PUNCT,
    # Numbers
    "0": NUMERIC,
    "123": NUMERIC,
    "9:30": NUMERIC,
    "12:45": NUMERIC,
    "34.5": NUMERIC,
    "0.05": NUMERIC,
    "16%": NUMERIC,
}


def test_twitter_unk():
    for handle, expected in tests.items():
        actual = twitter_unk(handle)
        assert expected == actual

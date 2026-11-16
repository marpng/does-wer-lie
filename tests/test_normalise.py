import pytest

from normalise import clean, normalisers, raw


def content(language: str, text: str) -> str:
    return normalisers(language)["content"](text)


def test_clean_removes_invisible_characters():
    assert clean("a​b") == "ab"


def test_raw_only_cleans_whitespace():
    assert raw("  Hello,   World ") == "Hello, World"


def test_every_language_has_a_content_normaliser():
    for language in ["en", "it", "de", "el", "gsw"]:
        assert "content" in normalisers(language)


def test_english_has_the_extra_whisper_normaliser():
    assert "standard_en" in normalisers("en")
    assert "standard_en" not in normalisers("el")


def test_english_keeps_bracketed_words_and_expands_units():
    assert content("en", "50 km/h (31 miles)") == "50 kilometers per hour 31 miles"


def test_standard_deletes_bracketed_words_but_content_keeps_them():
    text = "a city 50 kilometers (31 miles) away"
    assert "31" not in normalisers("en")["standard"](text)
    assert "31" in content("en", text)


@pytest.mark.parametrize("a,b", [("εξήντα", "60"), ("H Αθήνα", "Η Αθήνα")])
def test_greek_unifies_numbers_and_lookalike_letters(a, b):
    assert content("el", a) == content("el", b)


def test_greek_keeps_accents_because_they_change_the_word():
    assert content("el", "πότε") != content("el", "ποτέ")


@pytest.mark.parametrize("language", ["de", "gsw"])
def test_german_unifies_numbers_and_sharp_s(language):
    assert content(language, "dreihundertfünfzig Meter") == content(language, "350 Meter")
    assert content(language, "Straße") == content(language, "Strasse")


def test_italian_unifies_numbers_and_accent_direction():
    assert content("it", "trentotto") == content("it", "38")
    assert content("it", "perché") == content("it", "perchè")


def test_italian_keeps_e_with_and_without_accent_apart():
    assert content("it", "è") != content("it", "e")

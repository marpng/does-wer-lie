import pytest

from compute_metrics import error_counts


def test_identical_text_has_no_errors():
    counts = error_counts("the cat sat", "the cat sat")
    assert counts["word_errors"] == 0 and counts["char_errors"] == 0
    assert counts["ref_words"] == 3


def test_substitution_deletion_and_insertion_each_count_once():
    assert error_counts("the cat sat", "the dog sat")["word_errors"] == 1
    assert error_counts("the cat sat", "the cat")["word_errors"] == 1
    assert error_counts("the cat", "the big cat")["word_errors"] == 1


def test_reference_length_does_not_include_insertions():
    assert error_counts("a b", "a b c d")["ref_words"] == 2


def test_split_words_are_forgiven_only_on_request():
    assert error_counts("tuna fish", "tunafish")["word_errors"] == 2
    assert error_counts("tuna fish", "tunafish", forgive_splits=True)["word_errors"] == 0


def test_forgiving_splits_does_not_hide_real_errors():
    assert error_counts("tuna fish is good", "tunafish is bad", forgive_splits=True)["word_errors"] == 1


@pytest.mark.parametrize("hypothesis,errors", [("abc", 0), ("abd", 1), ("ab", 1), ("abcd", 1)])
def test_character_errors(hypothesis, errors):
    assert error_counts("abc", hypothesis)["char_errors"] == errors

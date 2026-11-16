import pytest

from analyse_error_types import classify, count_errors, edit_distance, strip_accents


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        ("Αθήνα.", "αθήνα", "punct_case"),
        ("μια", "μία", "accent"),
        ("10", "δέκα", "number"),
        ("Τζον", "John", "latin"),
        ("σπίτι", "σπίτια", "inflection"),
        ("φλοιός", "φλιός", "spelling"),
        ("τίγρεις", "πορτοκάλι", "other_word"),
    ],
)
def test_classify(reference, hypothesis, expected):
    assert classify(reference, hypothesis) == expected


def test_final_sigma_counts_as_the_same_letter_when_accents_are_ignored():
    assert strip_accents("λόγος") == strip_accents("λόγοσ")


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0


def test_count_errors_sorts_every_error_into_one_type():
    counts, ref_words, examples = count_errors("the cat sat", "the dog")
    assert ref_words == 3
    assert counts["other_word"] == 1 and counts["deleted"] == 1
    assert sum(counts.values()) == len(examples) == 2


def test_count_errors_on_identical_text_is_empty():
    counts, _, examples = count_errors("a b c", "a b c")
    assert sum(counts.values()) == 0 and examples == []

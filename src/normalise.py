"""Text normalisers for scoring
Reference and hypothesis always go through the same one

    raw          whitespace only
    standard     Whisper BasicTextNormalizer
    standard_en  Whisper EnglishTextNormalizer (English only)
    content      language rules that only remove differences a reader would not call an error

Content rules for all languages: words in brackets are kept
English: EnglishTextNormalizer plus unit abbreviations
Greek: accents and final sigma are kept
       number words digits and ordinals are unified
       a few abbreviations are expanded
       Latin look-alike letters inside Greek words are fixed
"""

import re
import unicodedata

from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

_basic = BasicTextNormalizer()
_english = EnglishTextNormalizer()


def clean(text: str) -> str:
    """Same Unicode form everywhere and no invisible format characters"""
    text = unicodedata.normalize("NFC", text)
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def raw(text: str) -> str:
    return " ".join(text.split())


def _unbracket(text: str) -> str:
    return re.sub(r"[()\[\]{}]", " ", text)


# ---------------------------------------------------------------- English

_EN_UNITS = [(r"(?<![a-z])km\s?/\s?h\b", " kilometers per hour"), (r"(?<![a-z])km\b", " kilometers")]


def content_en(text: str) -> str:
    text = _unbracket(text)
    for pattern, replacement in _EN_UNITS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return _english(text)


# ---------------------------------------------------------------- Greek

# Latin capitals that look like Greek ones, fixed only inside Greek words or as a lone article
_HOMOGLYPHS = str.maketrans("ABEHIKMNOPTXYZ", "ΑΒΕΗΙΚΜΝΟΡΤΧΥΖ")
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")

# Applied before lowercasing because "π.Χ." (BC) and "π.χ." (e.g.) differ only in case
_EL_ABBREVIATIONS = [
    (r"\bπ\.\s?Χ\.?", " προχριστού "), (r"\bπρο\s+Χριστού\b", " προχριστού "),
    (r"\bμ\.\s?Χ\.?", " μεταχριστόν "), (r"\bμετά\s+Χριστόν?\b", " μεταχριστόν "),
    (r"\bχλμ\b\.?", " χιλιόμετρα "), (r"\bχιλι(?:όμετρο|ομέτρου|ομέτρων)\b", " χιλιόμετρα "),
    (r"%", " τοιςεκατό "), (r"\bτοις\s+εκατό\b", " τοιςεκατό "),
]

_ONE = {"ένα", "ένας", "έναν", "μία", "μια", "ενός", "μιας"}  # also the indefinite article so only inside a number
_CARDINALS = {
    2: "δύο δυο", 3: "τρεις τρία τριών", 4: "τέσσερις τέσσερα τεσσάρων", 5: "πέντε", 6: "έξι",
    7: "επτά εφτά", 8: "οκτώ οχτώ", 9: "εννέα εννιά", 10: "δέκα", 11: "έντεκα ένδεκα", 12: "δώδεκα",
    13: "δεκατρείς δεκατρία δεκατριών", 14: "δεκατέσσερις δεκατέσσερα δεκατεσσάρων", 15: "δεκαπέντε",
    16: "δεκαέξι δεκάξι", 17: "δεκαεπτά δεκαεφτά", 18: "δεκαοκτώ δεκαοχτώ", 19: "δεκαεννέα δεκαεννιά",
    20: "είκοσι", 30: "τριάντα", 40: "σαράντα", 50: "πενήντα", 60: "εξήντα", 70: "εβδομήντα",
    80: "ογδόντα", 90: "ενενήντα", 100: "εκατό εκατόν", 1000: "χίλια χίλιοι χίλιες χίλιους χιλίων",
}
_HUNDREDS = {  # stressed stem and unstressed stem
    200: [("διακόσι", "διακοσ")], 300: [("τριακόσι", "τριακοσ")], 400: [("τετρακόσι", "τετρακοσ")],
    500: [("πεντακόσι", "πεντακοσ")], 600: [("εξακόσι", "εξακοσ")],
    700: [("επτακόσι", "επτακοσ"), ("εφτακόσι", "εφτακοσ")],
    800: [("οκτακόσι", "οκτακοσ"), ("οχτακόσι", "οχτακοσ")],
    900: [("εννιακόσι", "εννιακοσ"), ("εννεακόσι", "εννεακοσ")],
}
_MULTIPLIERS = {
    10**3: "χιλιάδες χιλιάδων",
    10**6: "εκατομμύριο εκατομμύρια εκατομμυρίου εκατομμυρίων",
    10**9: "δισεκατομμύριο δισεκατομμύρια δισεκατομμυρίου δισεκατομμυρίων",
}
_ORDINAL_STEMS = {  # stress on the stem
    1: "πρώτ", 2: "δεύτερ", 3: "τρίτ", 4: "τέταρτ", 5: "πέμπτ", 6: "έκτ", 7: "έβδομ", 8: "όγδο",
    9: "ένατ", 10: "δέκατ", 11: "ενδέκατ εντέκατ", 12: "δωδέκατ",
}
_ORDINAL_STEMS_FINAL = {  # stress on the ending
    20: "εικοστ", 30: "τριακοστ", 40: "τεσσαρακοστ", 50: "πεντηκοστ", 60: "εξηκοστ",
    70: "εβδομηκοστ", 80: "ογδοηκοστ", 90: "ενενηκοστ",
}
_ENDINGS = "ος ου ο ον η ης ην οι ων ους ες α".split()
_ENDINGS_STRESSED = "ός ού ό όν ή ής ήν οί ών ούς ές ά".split()

CARDINAL = {form: value for value, forms in _CARDINALS.items() for form in forms.split()}
for _value, _stems in _HUNDREDS.items():
    for _stressed, _unstressed in _stems:
        CARDINAL.update({_stressed + e: _value for e in ("α", "οι", "ες", "ους")})
        CARDINAL[_unstressed + "ίων"] = _value
MULTIPLIER = {form: value for value, forms in _MULTIPLIERS.items() for form in forms.split()}
ORDINAL = {stem + e: value for value, stems in _ORDINAL_STEMS.items() for stem in stems.split() for e in _ENDINGS}
ORDINAL.update({stem + e: value for value, stem in _ORDINAL_STEMS_FINAL.items() for e in _ENDINGS_STRESSED})
_ORDINAL_DIGITS = re.compile(r"^(\d+)(?:ος|ου|ον|ο|ης|ην|η|οι|ων|ους|ες)$")


def _fix_homoglyphs(text: str) -> str:
    def fix(match: re.Match) -> str:
        word = match.group()
        mapped = word.translate(_HOMOGLYPHS)
        lone_article = word in ("H", "O")
        return mapped if lone_article or (mapped != word and _GREEK.search(word) and not re.search(r"[A-Za-z]", mapped)) else word
    return re.sub(r"[^\W\d_]+", fix, text)


def _magnitude(value: int) -> int:
    return len(str(value))


def _numbers_to_digits(tokens: list[str]) -> list[str]:
    """Greek number words to digits ("δέκα χιλιάδες" to "10000") and ordinals to "<n>ο" ("δέκατη πέμπτη" to "15ο")"""
    out, i = [], 0
    while i < len(tokens):
        # ordinals, a tens ordinal may be followed by a units ordinal
        if tokens[i] in ORDINAL:
            value = ORDINAL[tokens[i]]
            if value >= 10 and value % 10 == 0 and i + 1 < len(tokens) and ORDINAL.get(tokens[i + 1], 99) < 10:
                value += ORDINAL[tokens[i + 1]]
                i += 1
            out.append(f"{value}ο")
            i += 1
            continue
        if match := _ORDINAL_DIGITS.match(tokens[i]):
            out.append(f"{match.group(1)}ο")
            i += 1
            continue

        # cardinals, add up a run of number words
        total, current, parts, j = 0, 0, 0, i
        while j < len(tokens):
            token = tokens[j]
            if token in MULTIPLIER and (current or token != tokens[i] or MULTIPLIER[token] >= 10**6):
                total += max(current, 1) * MULTIPLIER[token]
                current = 0
            elif token in CARDINAL and (not current or _magnitude(CARDINAL[token]) < _magnitude(current % 1000 or 1000)
                                        and current % 10 ** _magnitude(CARDINAL[token]) == 0):
                current += CARDINAL[token]
            elif token in _ONE and (current and current % 10 == 0 or j + 1 < len(tokens) and tokens[j + 1] in MULTIPLIER):
                current += 1
            elif token.isdigit() and not current and j + 1 < len(tokens) and tokens[j + 1] in MULTIPLIER:
                current = int(token)
            else:
                break
            parts += 1
            j += 1
        if parts and (j - i > 1 or tokens[i] in CARDINAL):
            out.append(str(total + current))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def content_el(text: str) -> str:
    text = _fix_homoglyphs(_unbracket(text))
    for pattern, replacement in _EL_ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)  # 10.000 -> 10000
    return " ".join(_numbers_to_digits(_basic(text).split()))


# ---------------------------------------------------------------- registry

CONTENT = {"en": content_en, "el": content_el}
# Languages where "tuna fish" and "tunafish" count as correct
# Not Greek because a moved word boundary there usually gives a different word
FORGIVE_SPLITS = {"en"}


def normalisers(language: str) -> dict:
    out = {"raw": raw, "standard": _basic}
    if language == "en":
        out["standard_en"] = _english
    if language in CONTENT:
        out["content"] = CONTENT[language]
    return out

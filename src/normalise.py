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
German (de, and gsw whose references are Standard German): ß and ss are the same
       number words digits and ordinals are unified ("dreihundertfünfzig" and "350", "10'000" and "10000", "dritte" and "3.")
       % and Prozent are the same and a few abbreviations are expanded
       compounds split differently are not errors ("Kasernen Areal" and "Kasernenareal")
Italian: acute and grave accents are not told apart ("perché" and "perchè") while "è" and "e" stay different
         number words digits and ordinals are unified ("trentotto" and "38", "XIX secolo" and "diciannovesimo secolo")
         % and per cento are the same and a few abbreviations are expanded
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


# ---------------------------------------------------------------- German (also Swiss German)

# Applied before lowercasing and before punctuation is removed. Both sides go through the same rules.
_DE_ABBREVIATIONS = [
    (r"\bz\.\s?B\.", " zum beispiel "), (r"\bd\.\s?h\.", " das heisst "), (r"\bu\.\s?a\.", " unter anderem "),
    (r"\bbzw\.", " beziehungsweise "), (r"\busw\.", " und so weiter "), (r"\betc\.", " und so weiter "),
    (r"\bca\.", " zirka "), (r"\bcirca\b", " zirka "), (r"\bNr\.", " nummer "), (r"\bDr\.", " doktor "),
    (r"\bProf\.", " professor "), (r"\bSt\.(?=\s+[A-ZÄÖÜ])", " sankt "),
    (r"\bMio\.", " millionen "), (r"\bMrd\.", " milliarden "),
    (r"(?<![\w])km\s?/\s?h\b", " kilometer pro stunde "), (r"(?<![\w])km\b", " kilometer "),
    (r"(?<![\w])kg\b", " kilogramm "), (r"\bCHF\b", " franken "), (r"\bFr\.(?=\s*\d)", " franken "),
    (r"%", " prozent "),
]
_MONTHS = "januar februar märz april mai juni juli august september oktober november dezember jänner".split()
_ORDINAL_NOUNS = set(_MONTHS) | {"jahrhundert", "jahrhunderts", "jahrtausend", "mal", "stock", "platz", "rang", "klasse", "runde", "liga"}

_DE_UNITS = {"eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9}
_DE_TEENS = {"zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
             "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
_DE_TENS = {"zwanzig": 20, "dreissig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90}
_DE_SCALES = {"million": 10**6, "millionen": 10**6, "milliarde": 10**9, "milliarden": 10**9}
_DE_ARTICLE_ONE = {"ein", "eine", "einen", "einer", "einem"}  # "1" only directly before Million/Milliarde
_DE_ORDINAL_SPECIAL = {"erst": 1, "zweit": 2, "dritt": 3, "siebt": 7, "siebent": 7, "acht": 8}


def _de_below_100(word: str) -> int | None:
    if word in _DE_UNITS:
        return _DE_UNITS[word]
    if word in _DE_TEENS:
        return _DE_TEENS[word]
    if word in _DE_TENS:
        return _DE_TENS[word]
    for tens, value in _DE_TENS.items():  # "einundzwanzig", "dreiundvierzig"
        if word.endswith("und" + tens):
            unit = word[: -len("und" + tens)]
            unit = "eins" if unit == "ein" else unit
            if unit in _DE_UNITS:
                return _DE_UNITS[unit] + value
    return None


def _de_below_1000(word: str) -> int | None:
    if word == "":
        return 0
    if "hundert" in word:
        head, _, tail = word.partition("hundert")
        if head in ("", "ein"):
            factor = 1
        else:
            factor = _de_below_100(head)  # also "neunzehnhundert" (years)
            if factor is None:
                return None
        rest = _de_below_1000_tail(tail)
        return None if rest is None else factor * 100 + rest
    return _de_below_100(word)


def _de_below_1000_tail(word: str) -> int | None:
    if word == "":
        return 0
    word = word[3:] if word.startswith("und") else word  # "hundertundzwei"
    return _de_below_100("eins" if word == "ein" else word)


def german_number(word: str) -> int | None:
    """One German number word -> its value ("zweitausendvierundzwanzig" -> 2024), else None. "ein" alone is not a number."""
    if "tausend" in word:
        head, _, tail = word.partition("tausend")
        factor = 1 if head in ("", "ein") else _de_below_1000(head)
        rest = _de_below_1000(tail[3:] if tail.startswith("und") else tail)
        if factor is None or rest is None:
            return None
        return factor * 1000 + rest
    return _de_below_1000(word) if word not in ("", "ein") else None


def german_ordinal(word: str) -> int | None:
    """"dritten" -> 3, "zwanzigste" -> 20, "einundzwanzigsten" -> 21."""
    match = re.fullmatch(r"(\w+?)(?:e|en|er|es|em)", word)
    if not match:
        return None
    stem = match.group(1)
    if stem in _DE_ORDINAL_SPECIAL:
        return _DE_ORDINAL_SPECIAL[stem]
    for prefix_end in ("st", "t"):
        if stem.endswith(prefix_end):
            value = german_number(stem[: -len(prefix_end)])
            if value is not None and (prefix_end == "st") == (value >= 20 and value not in _DE_TEENS.values()):
                return value
    return None


def _de_numbers(tokens: list[str]) -> list[str]:
    out = []
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in _DE_ARTICLE_ONE and nxt in _DE_SCALES:
            out.append("1")
        elif (value := german_ordinal(token)) is not None:
            out.append(f"{value}.")
        elif (value := german_number(token)) is not None:
            out.append(str(value))
        else:
            out.append(token)
    return out


def content_de(text: str) -> str:
    text = _unbracket(text).replace("ß", "ss").replace("ẞ", "SS")
    for pattern, replacement in _DE_ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"(?<=\d)['’.   ](?=\d{3}\b)", "", text)  # 10'000 / 10.000 / 10 000 -> 10000
    text = re.sub(r"(?<=\d),(?=\d)", " komma ", text)                  # 2,5 -> 2 komma 5
    # "3. Mai", "20. Jahrhundert": a digit with a full stop before a lowercase word or a date/ranking noun is an ordinal
    text = re.sub(r"\b(\d+)\.(?=\s+(\w+))",
                  lambda m: f"{m.group(1)}ORD " if m.group(2)[0].islower() or m.group(2).lower() in _ORDINAL_NOUNS else m.group(0),
                  text)
    tokens = _basic(text.replace("ORD ", "ord ")).split()
    tokens = [t[:-3] + "." if re.fullmatch(r"\d+ord", t) else t for t in tokens]
    return " ".join(_de_numbers(tokens))


# ---------------------------------------------------------------- Italian

_IT_ACCENTS = str.maketrans("áéíóúÁÉÍÓÚ", "àèìòùÀÈÌÒÙ")  # direction of the accent only; "è" vs "e" stays
_IT_ABBREVIATIONS = [
    (r"(?<![\w])km\s?/\s?h\b", " chilometri orari "), (r"(?<![\w])km\b", " chilometri "), (r"(?<![\w])kg\b", " chilogrammi "),
    (r"\bchilometri all'ora\b", " chilometri orari "), (r"\becc\.", " eccetera "), (r"\betc\.", " eccetera "),
    (r"\ba\.\s?C\.", " avanticristo "), (r"\bavanti Cristo\b", " avanticristo "),
    (r"\bd\.\s?C\.", " dopocristo "), (r"\bdopo Cristo\b", " dopocristo "),
    (r"\bdott\.", " dottor "), (r"\bdottore\b", " dottor "), (r"\bsig\.", " signor "), (r"\bsignore\b(?=\s+[A-Z])", " signor "),
    (r"%", " percento "), (r"\bper\s+cento\b", " percento "),
]
_IT_UNITS = {"uno": 1, "due": 2, "trè": 3, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9}
_IT_TEENS = {"dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15, "sedici": 16,
             "diciassette": 17, "diciotto": 18, "diciannove": 19}
_IT_TENS = {"venti": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50, "sessanta": 60, "settanta": 70, "ottanta": 80, "novanta": 90}
_IT_SCALES = {"milione", "milioni", "miliardo", "miliardi"}
_IT_ARTICLE_ONE = {"un", "uno", "una"}  # "1" only directly before milione/miliardo
_IT_ORDINALS = {"prim": 1, "second": 2, "terz": 3, "quart": 4, "quint": 5, "sest": 6, "settim": 7, "ottav": 8, "non": 9, "decim": 10}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _it_below_100(word: str) -> int | None:
    for table in (_IT_UNITS, _IT_TEENS, _IT_TENS):
        if word in table:
            return table[word]
    for tens, value in _IT_TENS.items():
        for stem in (tens, tens[:-1]):  # "ventidue", but "ventuno", "ventotto" drop the vowel
            if word.startswith(stem):
                unit = word[len(stem):]
                unit = "uno" if unit == "un" else unit
                if unit in _IT_UNITS and (stem == tens) != (unit in ("uno", "otto")):
                    return value + _IT_UNITS[unit]
    return None


def _it_below_1000(word: str) -> int | None:
    if word == "":
        return 0
    if "cent" in word:
        head, _, tail = word.partition("cent")
        factor = 1 if head == "" else _IT_UNITS.get(head)
        if factor is None or head in ("uno", "trè"):
            return None
        if tail in ("o", ""):
            rest = 0
        else:  # "centodue", "centottanta" (the o of cento dropped before otto/ottanta)
            rest = _it_below_100(tail[1:]) if tail.startswith("o") else None
            rest = _it_below_100(tail) if rest is None else rest
        return None if rest is None else factor * 100 + rest
    return _it_below_100(word)


def italian_number(word: str) -> int | None:
    """One Italian number word -> value ("duemilaventisei" -> 2026), else None. un/uno/una alone are not numbers."""
    if word in _IT_ARTICLE_ONE or word == "":
        return None
    if word.startswith("mille"):
        rest = _it_below_1000(word[5:])
        return None if rest is None else 1000 + rest
    if "mila" in word:
        head, _, tail = word.partition("mila")
        factor, rest = _it_below_1000(head), _it_below_1000(tail)
        return None if factor in (None, 0, 1) or rest is None else factor * 1000 + rest
    return _it_below_1000(word)


def italian_ordinal(word: str) -> int | None:
    """"terza" -> 3, "ventesimo" -> 20, "diciannovesimi" -> 19."""
    match = re.fullmatch(r"(\w+?)[oaie]", word)
    if not match:
        return None
    stem = match.group(1)
    if stem in _IT_ORDINALS:
        return _IT_ORDINALS[stem]
    if stem.endswith("esim"):
        base = stem[:-4]
        for vowel in ("", "i", "o", "e", "a"):  # "undic|esimo" <- undici, "vent|esimo" <- venti, "mill|esimo" <- mille
            value = italian_number(base + vowel)
            if value is not None and value > 10:
                return value
    return None


def _roman(text: str) -> int | None:
    values = [_ROMAN[c] for c in text]
    total = sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))
    return total if 0 < total < 4000 else None


def _it_numbers(tokens: list[str]) -> list[str]:
    out = []
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in _IT_ARTICLE_ONE and nxt in _IT_SCALES:
            out.append("1")
        elif (value := italian_ordinal(token)) is not None:
            out.append(f"{value}.")
        elif (value := italian_number(token)) is not None:
            out.append(str(value))
        else:
            out.append(token)
    return out


def content_it(text: str) -> str:
    text = _unbracket(text).translate(_IT_ACCENTS)
    for pattern, replacement in _IT_ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)['’.   ](?=\d{3}\b)", "", text)   # 10.000 -> 10000
    text = re.sub(r"(?<=\d),(?=\d)", " virgola ", text)                 # 2,5 -> 2 virgola 5
    text = re.sub(r"\b(\d+)\s?[º°ª]", r" \1ORD ", text)                 # 1º, 2ª
    text = re.sub(r"\b([IVXLCDM]+)(?=\s+secol)", lambda m: f" {_roman(m.group(1))}ORD " if _roman(m.group(1)) else m.group(0), text)
    tokens = [t[:-3] + "." if re.fullmatch(r"\d+ord", t) else t for t in _basic(text).split()]
    return " ".join(_it_numbers(tokens))


# ---------------------------------------------------------------- registry

CONTENT = {"en": content_en, "el": content_el, "de": content_de, "gsw": content_de, "it": content_it}
# Languages where "tuna fish" and "tunafish" count as correct
# Not Greek because a moved word boundary there usually gives a different word
# German and Swiss German compounds written apart or together ("Kasernen Areal" and "Kasernenareal") keep the meaning
FORGIVE_SPLITS = {"en", "de", "gsw"}


def normalisers(language: str) -> dict:
    out = {"raw": raw, "standard": _basic}
    if language == "en":
        out["standard_en"] = _english
    if language in CONTENT:
        out["content"] = CONTENT[language]
    return out

"""Text for synthetic pages: real English sentences plus the tokens books lack.

The previous generator cycled one 75-word paragraph, so the recognizer learned
word shapes rather than letters and had never seen a digit. This draws from a
20k-sentence public-domain corpus (ocr/data/corpus/english.txt) and splices in
numbers, dates, amounts, codes, initials and shouty words at a controlled rate,
so every character in the charset appears in realistic contexts.
"""

from __future__ import annotations

import random
import string
from functools import lru_cache
from pathlib import Path

from ocr.config import DEFAULT_CHARSET

CORPUS_PATH = Path(__file__).resolve().parent / "corpus" / "english.txt"
_ALLOWED = set(DEFAULT_CHARSET)

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
UNITS = ["kg", "cm", "mm", "km", "lb", "oz", "GB", "MB", "kHz", "MHz", "pp", "vol", "no", "ed", "Fig", "Eq", "Tab"]
FIRST = ["John", "Mary", "James", "Anna", "Robert", "Elena", "Michael", "Sara", "David", "Laura", "Thomas", "Maria", "Peter", "Julia", "George", "Alice"]
LAST = ["Smith", "Johnson", "Brown", "Garcia", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Clark", "Lewis", "Walker", "Hall", "Young", "King", "Wright"]
ACRONYMS = ["OCR", "PDF", "CPU", "GPU", "USA", "UK", "EU", "NASA", "IEEE", "ACM", "ISBN", "DOI", "URL", "API", "ID", "TBD", "FAQ", "PhD", "MSc", "Inc", "Ltd", "Co", "Dept", "Univ"]


@lru_cache(maxsize=1)
def load_sentences(path: Path = CORPUS_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"text corpus missing: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def sanitize(text: str) -> str:
    """Only characters the recognizer can emit; the label must match the pixels exactly."""
    return "".join(c for c in text if c in _ALLOWED)


def _number() -> str:
    r = random.random()
    if r < 0.30:
        return str(random.randint(0, 9999))
    if r < 0.45:
        return f"{random.randint(1, 999)},{random.randint(0, 999):03d}"
    if r < 0.60:
        return f"{random.uniform(0, 1000):.{random.choice([1, 2, 3])}f}"
    if r < 0.70:
        return f"{random.randint(1, 100)}%"
    if r < 0.80:
        return f"${random.randint(1, 9999)}.{random.randint(0, 99):02d}"
    if r < 0.90:
        return f"{random.randint(1, 500)} {random.choice(UNITS)}"
    return f"({random.randint(1, 60)})"


def _date() -> str:
    d, m, y = random.randint(1, 28), random.randint(1, 12), random.randint(1890, 2030)
    return random.choice([
        f"{MONTHS[m - 1]} {d}, {y}",
        f"{d} {MONTHS_SHORT[m - 1]} {y}",
        f"{m:02d}/{d:02d}/{y}",
        f"{y}-{m:02d}-{d:02d}",
        f"{d}.{m}.{y}",
        f"{MONTHS[m - 1]} {y}",
    ])


def _code() -> str:
    letters = "".join(random.choices(string.ascii_uppercase, k=random.randint(2, 4)))
    digits = "".join(random.choices(string.digits, k=random.randint(3, 6)))
    return random.choice([
        f"{letters}-{digits}",
        f"{letters}{digits}",
        f"No. {random.randint(1, 9999)}",
        f"Ref: {letters}/{digits}",
        f"#{digits}",
        f"{random.randint(1, 99)}.{random.randint(1, 20)}.{random.randint(1, 9)}",
        f"p. {random.randint(1, 600)}",
        f"{random.choice(['Tel', 'Fax'])}: {random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
    ])


def _name() -> str:
    r = random.random()
    if r < 0.4:
        return f"{random.choice(FIRST)} {random.choice(LAST)}"
    if r < 0.7:
        return f"{random.choice(['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Ms.'])} {random.choice(LAST)}"
    return f"{random.choice(string.ascii_uppercase)}. {random.choice(LAST)}"


def _shouty() -> str:
    return random.choice(ACRONYMS) if random.random() < 0.6 else random.choice(LAST).upper()


def _symbols() -> str:
    """The rare charset members: email, math, paths, code-ish fragments."""
    w = random.choice(LAST).lower()
    return random.choice([
        f"{random.choice(FIRST).lower()}.{w}@{random.choice(['mail', 'univ', 'example', 'lab'])}.{random.choice(['com', 'edu', 'org'])}",
        f"{random.randint(1, 99)} + {random.randint(1, 99)} = {random.randint(2, 198)}",
        f"x <= {random.randint(1, 50)}",
        f"{random.randint(1, 9)} > {random.randint(1, 9)}",
        f"~/{w}/{random.choice(['docs', 'data', 'src'])}",
        f"C:\\{w.capitalize()}\\{random.choice(['Temp', 'Files', 'Old'])}",
        f"{{{w}}}",
        f"{w}_{random.randint(1, 9)}",
        f"x^{random.randint(2, 9)}",
        f"{w}|{random.choice(LAST).lower()}",
        f"`{w}`",
        f"[{random.randint(1, 40)}]",
        f"{w} & {random.choice(LAST).lower()}",
        f"{random.randint(1, 9)}*{random.randint(1, 9)}",
    ])


def special_token() -> str:
    return random.choices([_number, _date, _code, _name, _shouty, _symbols], weights=[0.32, 0.18, 0.18, 0.14, 0.08, 0.10])[0]()


def paragraph_words(special_rate: float = 0.06) -> list[str]:
    """A run of 4-7 real sentences as a word list, with special tokens spliced in.

    Roughly one word in sixteen becomes a number, date, code, name or acronym,
    which is far denser than prose but keeps every such form common enough
    for the recognizer to learn it.
    """
    sentences = load_sentences()
    words = " ".join(random.choice(sentences) for _ in range(random.randint(4, 7))).split()
    out: list[str] = []
    for word in words:
        if random.random() < special_rate:
            token = sanitize(special_token())
            # keep trailing punctuation of the word it replaces, so lines still end naturally
            trail = word[-1] if word[-1] in ",.;:!?" else ""
            out.append(token + trail)
        else:
            out.append(word)
    return out

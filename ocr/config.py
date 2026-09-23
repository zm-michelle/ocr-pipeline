from __future__ import annotations

import os
import string
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PNG_EXTENSION = ".png"

# Keep the recognizer literal: case, punctuation, and spaces are preserved.
DEFAULT_CHARSET = (
    string.ascii_uppercase
    + string.ascii_lowercase
    + string.digits
    + " "
    + r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
)

CTC_BLANK_INDEX = 0

# Every face that matches any name below is a candidate; the generator picks one
# at random per page (see FontResolver.pick). Names cover macOS system fonts,
# the DejaVu/Liberation/FreeFont set remote/fonts.sh installs,
# and what RHEL/Ubuntu ship system-wide (URW base35, Droid, Cantarell).
FONT_FAMILIES = {
    "typewriter": [
        "Courier New", "Courier", "Liberation Mono", "DejaVu Sans Mono", "Nimbus Mono PS",
        "Droid Sans Mono", "FreeMono", "Menlo", "Monaco", "Andale Mono", "PT Mono",
    ],
    "sans": [
        "Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "Nimbus Sans", "Droid Sans",
        "FreeSans", "Cantarell", "URW Gothic", "Verdana", "Trebuchet MS", "Tahoma",
        "Avenir", "Gill Sans", "Optima", "Futura",
    ],
    "serif": [
        "Times New Roman", "Times", "DejaVu Serif", "Liberation Serif", "Nimbus Roman", "Droid Serif",
        "FreeSerif", "URW Bookman", "C059", "P052", "Georgia", "Palatino", "Baskerville",
        "Book Antiqua", "Charter", "Hoefler Text", "Didot",
    ],
}

# Substrings (matched against the lower-cased file name with -/_ turned into spaces)
# that identify each name. Keep them specific enough not to catch symbol or CJK faces.
FONT_FILE_HINTS = {
    "Courier New": ["courier new", "cour.ttf", "courbd", "couri", "courbi"],
    "Courier": ["courier.ttc", "courier.ttf"],
    "Liberation Mono": ["liberationmono", "liberation mono"],
    "DejaVu Sans Mono": ["dejavusansmono", "dejavu sans mono"],
    "Nimbus Mono PS": ["nimbusmonops", "nimbus mono"],
    "Droid Sans Mono": ["droidsansmono"],
    "FreeMono": ["freemono"],
    "Menlo": ["menlo"],
    "Monaco": ["monaco"],
    "Andale Mono": ["andale mono", "andalemono"],
    "PT Mono": ["ptmono", "pt mono"],
    "Arial": ["arial.ttf", "arialbd", "ariali", "arialbi", "arial bold", "arial italic", "arial narrow"],
    "Helvetica": ["helvetica.ttc", "helvetica.ttf", "helveticaneue", "helvetica neue"],
    "DejaVu Sans": ["dejavusans.ttf", "dejavusans bold", "dejavusans oblique", "dejavusans boldoblique", "dejavu sans.ttf"],
    "Liberation Sans": ["liberationsans", "liberation sans"],
    "Nimbus Sans": ["nimbussans", "nimbus sans"],
    "Droid Sans": ["droidsans.ttf", "droidsans bold.ttf"],
    "FreeSans": ["freesans"],
    "Cantarell": ["cantarell"],
    "URW Gothic": ["urwgothic", "urw gothic"],
    "Verdana": ["verdana"],
    "Trebuchet MS": ["trebuchet"],
    "Tahoma": ["tahoma"],
    "Avenir": ["avenir.ttc", "avenir next.ttc"],
    "Gill Sans": ["gillsans", "gill sans"],
    "Optima": ["optima"],
    "Futura": ["futura"],
    "Times New Roman": ["times new roman", "timesnewroman", "times.ttf", "timesbd", "timesi", "timesbi"],
    "Times": ["times.ttc"],
    "DejaVu Serif": ["dejavuserif.ttf", "dejavuserif bold", "dejavuserif italic", "dejavuserif bolditalic", "dejavu serif.ttf"],
    "Liberation Serif": ["liberationserif", "liberation serif"],
    "Nimbus Roman": ["nimbusroman", "nimbus roman"],
    "Droid Serif": ["droidserif"],
    "FreeSerif": ["freeserif"],
    "URW Bookman": ["urwbookman", "urw bookman"],
    "C059": ["c059"],
    "P052": ["p052"],
    "Georgia": ["georgia.ttf", "georgia bold", "georgia italic"],
    "Palatino": ["palatino"],
    "Baskerville": ["baskerville"],
    "Book Antiqua": ["book antiqua", "bookantiqua"],
    "Charter": ["charter"],
    "Hoefler Text": ["hoefler text.ttc"],
    "Didot": ["didot"],
}

# Faces never used for Latin text, whatever family hint they happen to match.
FONT_FILE_BLOCKLIST = (
    "ornament", "symbol", "dingbat", "emoji", "fallback", "arabic", "hebrew", "georgian",
    "armenian", "devanagari", "thai", "ethiopic", "tamil", "japanese", "cjk", "unicode",
    "math", "braille", "wingding", "webding",
)

COMMON_FONT_DIRS = [
    *(Path(d) for d in os.environ.get("OCR_FONT_DIRS", "").split(os.pathsep) if d),
    Path.home() / ".fonts",
    Path.home() / ".local" / "share" / "fonts",
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
]

DEFAULT_PAGE_SIZE = (540, 258)  # width, height; matches SimulatedNoisyOffice patches
DEFAULT_DETECTOR_SIZE = (540, 258)  # width, height
DEFAULT_RECOGNIZER_HEIGHT = 32
# A 505x19 line scaled to height 32 is ~800 px wide; the old 512 canvas squashed
# 97% of lines (median 1.57x). 1280 fits essentially all of them, giving CTC 320
# time-steps. Checkpoints record their own size, so older 512-wide models still work.
DEFAULT_RECOGNIZER_WIDTH = 1280
LEGACY_RECOGNIZER_SIZE = (32, 512)  # for checkpoints that predate the size metadata

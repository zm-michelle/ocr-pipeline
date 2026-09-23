#!/usr/bin/env bash
# Install the fonts the synthetic generator needs WITHOUT root, into
# ~/.local/share/fonts/ocr (a directory ocr/config.py already searches).
#
# DejaVu Sans / Serif / Mono are pulled out of the matplotlib wheel, so the only
# thing this needs is PyPI access. Liberation (metric clones of Arial / Times /
# Courier) is fetched from its GitHub release; if that host is unreachable from
# the cluster we carry on -- DejaVu alone covers all three font families.
#
#   remote/fonts.sh            # uses `python` on PATH (activate the venv first)
#   PYTHON=python3.12 remote/fonts.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEST="${OCR_FONT_DEST:-$HOME/.local/share/fonts/ocr}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$DEST"

echo "-> DejaVu (from the matplotlib wheel)"
"$PYTHON" -m pip download -q --no-deps --dest "$WORK" matplotlib
WHEEL="$(ls "$WORK"/matplotlib-*.whl | head -1)"
"$PYTHON" - "$WHEEL" "$DEST" <<'PY'
import sys, zipfile, pathlib
wheel, dest = sys.argv[1], pathlib.Path(sys.argv[2])
n = 0
with zipfile.ZipFile(wheel) as z:
    for member in z.namelist():
        name = pathlib.Path(member).name
        if "/mpl-data/fonts/ttf/" in member and name.startswith("DejaVu") and name.endswith(".ttf"):
            (dest / name).write_bytes(z.read(member)); n += 1
print(f"   {n} DejaVu files")
PY

echo "-> Liberation (from GitHub; optional)"
LIB_URL="https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz"
if curl -fsSL --max-time 60 "$LIB_URL" -o "$WORK/liberation.tar.gz" 2>/dev/null; then
  tar -xzf "$WORK/liberation.tar.gz" -C "$WORK"
  n=0; for f in "$WORK"/liberation-fonts-ttf-*/*.ttf; do cp "$f" "$DEST/"; n=$((n+1)); done
  echo "   $n Liberation files"
else
  echo "   skipped (no access to github.com from here); DejaVu is enough"
fi

# More variety, all optional: the generator picks a random face per page, so
# every extra family here is a different look the recognizer learns to read.
fetch_tar() {  # name url glob-inside-archive
  echo "-> $1 (optional)"
  if curl -fsSL --max-time 90 "$2" -o "$WORK/$1.archive" 2>/dev/null; then
    mkdir -p "$WORK/$1" && ( cd "$WORK/$1" && { tar -xzf "../$1.archive" 2>/dev/null || unzip -qo "../$1.archive"; } )
    n=0; while IFS= read -r f; do cp "$f" "$DEST/"; n=$((n+1)); done < <(find "$WORK/$1" -type f \( -name "*.ttf" -o -name "*.otf" \) | grep -E "$3")
    echo "   $n files"
  else
    echo "   skipped (download failed)"
  fi
}
fetch_file() {  # name url
  echo "-> $1 (optional)"
  if curl -fsSL --max-time 60 "$2" -o "$DEST/$(basename "$2")" 2>/dev/null; then echo "   1 file"; else echo "   skipped"; fi
}

fetch_tar freefont "https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip" "Free(Sans|Serif|Mono)"
# Carlito/Caladea deliberately left out: their hinting makes FreeType ~10x slower
# per line than DejaVu, which dominated generation time for two faces' worth of variety.
# URW base35: RHEL ships it system-wide (/usr/share/fonts/urw-base35); fetch only if absent.
if ! ls /usr/share/fonts/urw-base35/*.otf >/dev/null 2>&1; then
  fetch_tar urw "https://github.com/ArtifexSoftware/urw-base35-fonts/archive/refs/tags/20200910.tar.gz" "(NimbusSans|NimbusRoman|NimbusMonoPS|URWBookman|URWGothic|C059|P052)-.*\.ttf$"
else
  echo "-> URW base35 already installed system-wide"
fi

echo "-> installed into $DEST:"
ls "$DEST" | sed 's/^/   /'
echo "-> checking the generator can see them"
"$PYTHON" -c "from ocr.data.synthetic import FontResolver; FontResolver().require_real_fonts()"

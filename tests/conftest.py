"""Put the repo root on sys.path so `import ocr` works without installing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

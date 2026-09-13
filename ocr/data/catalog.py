"""Discover browsable image collections on disk, with ground truth where a manifest exists.

The web app uses this to let you pick samples straight out of the datasets instead of
uploading a file. Listings and manifests are read lazily and cached, so pointing at a
directory with 40k line crops costs nothing until you actually open it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ocr.config import IMAGE_EXTENSIONS

MAX_DEPTH = 4
SKIP_DIR_NAMES = {"__pycache__", ".git", ".ipynb_checkpoints"}

# Filename tokens that mark which *version* of a page an image is, rather than
# which page it is. NoisyOffice names its pairs FontLre_Noisec_TE / FontLre_Clean_TE,
# so dropping these tokens leaves a key shared by every version of one page.
VERSION_TOKEN = re.compile(r"^(noise\w*|clean|degraded|noisy|gt|groundtruth)$", re.IGNORECASE)


@dataclass(frozen=True)
class Collection:
    """One directory of images the UI can browse."""

    key: str
    label: str
    path: Path
    manifest_root: Path | None


class GroundTruth:
    """Image path -> expected transcript, read from a dataset's manifests.

    Keys are paths relative to the manifest root, matching how the manifests
    reference their own images (``line_crops/x.png``, ``pages/x.png``).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._texts: dict[str, str] | None = None

    @staticmethod
    def _items(data: object, *keys: str) -> list[dict]:
        if isinstance(data, dict):
            for key in keys:
                if data.get(key):
                    return data[key]
            return []
        return data if isinstance(data, list) else []

    def _build(self) -> dict[str, str]:
        texts: dict[str, str] = {}

        for manifest in sorted(self.root.glob("*manifest*.json")):
            try:
                data = json.loads(manifest.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            for item in self._items(data, "samples", "items", "labels"):
                image = item.get("image") or item.get("image_path")
                if image and item.get("text") is not None:
                    texts.setdefault(str(image), str(item["text"]))

            for page in self._items(data, "pages"):
                image = page.get("image") or page.get("image_path")
                lines = [line.get("text", "") for line in page.get("lines", [])]
                if image and lines:
                    texts.setdefault(str(image), "\n".join(lines))

        return texts

    def text_for(self, relative_path: str) -> str | None:
        if self._texts is None:
            self._texts = self._build()
        return self._texts.get(relative_path)


class DatasetCatalog:
    """Image collections discovered under one or more root directories."""

    def __init__(
        self,
        roots: list[str | Path],
        base: str | Path | None = None,
        max_depth: int = MAX_DEPTH,
        extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    ) -> None:
        self.base = Path(base).resolve() if base else Path.cwd().resolve()
        self.extensions = extensions
        self._collections: dict[str, Collection] = {}
        self._listings: dict[str, list[str]] = {}
        self._ground_truth: dict[Path, GroundTruth] = {}
        self._pair_indexes: dict[str, dict[tuple[str, ...], list[str]]] = {}

        for root in roots:
            root = Path(root)
            if root.is_dir():
                self._discover(root.resolve(), max_depth)

    # -- discovery -------------------------------------------------------

    def _key_for(self, path: Path) -> str:
        try:
            return path.relative_to(self.base).as_posix()
        except ValueError:
            return path.as_posix()

    def _has_images(self, path: Path) -> bool:
        try:
            return any(e.is_file() and Path(e.name).suffix.lower() in self.extensions for e in path.iterdir())
        except OSError:
            return False

    def _nearest_manifest_root(self, path: Path, stop: Path) -> Path | None:
        current = path
        while True:
            if any(current.glob("*manifest*.json")):
                return current
            if current == stop or current.parent == current:
                return None
            current = current.parent

    def _discover(self, root: Path, max_depth: int) -> None:
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if self._has_images(path):
                key = self._key_for(path)
                self._collections[key] = Collection(
                    key=key,
                    label=key,
                    path=path,
                    manifest_root=self._nearest_manifest_root(path, root),
                )
            if depth >= max_depth:
                continue
            try:
                children = [e for e in path.iterdir() if e.is_dir() and e.name not in SKIP_DIR_NAMES and not e.name.startswith(".")]
            except OSError:
                continue
            stack.extend((child, depth + 1) for child in children)

    # -- access ----------------------------------------------------------

    def collections(self) -> list[Collection]:
        return [self._collections[key] for key in sorted(self._collections)]

    def get(self, key: str) -> Collection | None:
        return self._collections.get(key)

    def images(self, key: str) -> list[str]:
        """Sorted file names directly inside the collection. Cached after the first call."""
        if key not in self._listings:
            collection = self._collections.get(key)
            if collection is None:
                return []
            try:
                names = [
                    entry.name
                    for entry in collection.path.iterdir()
                    if entry.is_file() and Path(entry.name).suffix.lower() in self.extensions
                ]
            except OSError:
                names = []
            self._listings[key] = sorted(names)
        return self._listings[key]

    def resolve(self, key: str, name: str) -> Path | None:
        """Map (collection, file name) to a path, or None if it is not a listed image.

        Membership in the cached listing is the traversal guard: anything with a
        separator or a `..` component simply is not in the list.
        """
        collection = self._collections.get(key)
        if collection is None or name not in set(self.images(key)):
            return None
        return collection.path / name

    # -- pairing ---------------------------------------------------------

    @staticmethod
    def _pair_key(name: str) -> tuple[str, ...] | None:
        """Key shared by every version of the same underlying page, or None.

        ``FontLre_Noisec_TE.png`` and ``FontLre_Clean_TE.png`` both reduce to
        ``("FontLre", "TE")``. A name carrying no version token is not pairable.
        """
        parts = Path(name).stem.split("_")
        kept = [p for p in parts if not VERSION_TOKEN.match(p)]
        if len(kept) == len(parts) or not kept:
            return None
        return tuple(kept)

    @staticmethod
    def _version_token(name: str) -> str:
        """The dropped token, e.g. ``Noisec`` — what makes this version distinct."""
        tokens = [p for p in Path(name).stem.split("_") if VERSION_TOKEN.match(p)]
        return " ".join(tokens)

    def _pair_index(self, key: str) -> dict[tuple[str, ...], list[str]]:
        if key not in self._pair_indexes:
            index: dict[tuple[str, ...], list[str]] = {}
            for name in self.images(key):
                pair_key = self._pair_key(name)
                if pair_key is not None:
                    index.setdefault(pair_key, []).append(name)
            self._pair_indexes[key] = index
        return self._pair_indexes[key]

    def counterparts(self, key: str, name: str) -> list[dict[str, str]]:
        """Other versions of the same page, found in sibling collections.

        For NoisyOffice this pairs a noisy scan with its clean originals, so the
        UI can offer them side by side.
        """
        collection = self._collections.get(key)
        pair_key = self._pair_key(name)
        if collection is None or pair_key is None:
            return []

        found: list[dict[str, str]] = []
        for other in self.collections():
            if other.key == key or other.path.parent != collection.path.parent:
                continue
            for match in self._pair_index(other.key).get(pair_key, []):
                found.append(
                    {
                        "dataset": other.key,
                        "name": match,
                        # Either half can be the distinguishing one: three clean
                        # collections share a filename, four noise variants share
                        # a directory. Carry both so the UI is never ambiguous.
                        "version": self._version_token(match),
                        "label": other.path.name,
                    }
                )
        return found

    def ground_truth_for(self, key: str, name: str) -> str | None:
        collection = self._collections.get(key)
        if collection is None or collection.manifest_root is None:
            return None
        path = self.resolve(key, name)
        if path is None:
            return None
        index = self._ground_truth.get(collection.manifest_root)
        if index is None:
            index = GroundTruth(collection.manifest_root)
            self._ground_truth[collection.manifest_root] = index
        try:
            relative = path.relative_to(collection.manifest_root).as_posix()
        except ValueError:
            return None
        return index.text_for(relative)

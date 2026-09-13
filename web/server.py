"""FastAPI app: drop an image in the browser, get the predicted text back.

Run it with:

    python -m web.server --recognizer_ckpt checkpoints/recognizer_last.pt \
                         --detector_ckpt checkpoints/detector_last.pt

The models are loaded once at startup and reused for every request. Images can be
uploaded, or picked straight out of the datasets under --samples_dir; where a dataset
ships a manifest, the expected transcript and live CER/WER come back with the result.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from ocr.config import IMAGE_EXTENSIONS
from ocr.data.catalog import DatasetCatalog
from ocr.evaluation import compute_cer, compute_wer
from ocr.inference import OCRPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_RENDER_SIDE = 2000  # cap what we send back to the browser, not what we OCR
THUMB_SIDE = 260
PAGE_SIZE_LIMIT = 200


class ModelStore:
    """Holds the pipeline plus the paths it was built from."""

    def __init__(self) -> None:
        self.pipeline: OCRPipeline | None = None
        self.recognizer_ckpt: Path | None = None
        self.detector_ckpt: Path | None = None
        self.error: str | None = None

    def load(self, recognizer_ckpt: str | Path, detector_ckpt: str | Path | None, device: str) -> None:
        recognizer_ckpt = Path(recognizer_ckpt)
        if not recognizer_ckpt.exists():
            self.error = f"recognizer checkpoint not found: {recognizer_ckpt}"
            return
        if detector_ckpt and not Path(detector_ckpt).exists():
            self.error = f"detector checkpoint not found: {detector_ckpt}"
            return
        try:
            self.pipeline = OCRPipeline.load(
                recognizer_ckpt=recognizer_ckpt,
                detector_ckpt=detector_ckpt,
                device=device,
                split_lines_without_detector=True,
            )
        except Exception as exc:  # surfaced in /api/health instead of killing the server
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self.recognizer_ckpt = recognizer_ckpt
        self.detector_ckpt = Path(detector_ckpt) if detector_ckpt else None
        self.error = None

    def require(self) -> OCRPipeline:
        if self.pipeline is None:
            raise HTTPException(status_code=503, detail=self.error or "models are not loaded")
        return self.pipeline


store = ModelStore()
catalog = DatasetCatalog(roots=[])
app = FastAPI(title="Degraded Document OCR", version="0.3.0")


def _png_data_uri(image: Image.Image, max_side: int = MAX_RENDER_SIDE) -> str:
    render = image.copy()
    render.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    render.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache(maxsize=1024)
def _thumbnail_png(path_str: str, mtime: float, size: int) -> bytes:
    """Cached by (path, mtime, size) so an edited file re-renders but a re-view does not."""
    with Image.open(path_str) as image:
        thumb = image.convert("L")
        thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
        buffer = io.BytesIO()
        thumb.save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/api/health")
def health() -> JSONResponse:
    pipeline = store.pipeline
    return JSONResponse(
        {
            "ready": pipeline is not None,
            "error": store.error,
            "device": str(pipeline.device) if pipeline else None,
            "detector": str(store.detector_ckpt) if store.detector_ckpt else None,
            "recognizer": str(store.recognizer_ckpt) if store.recognizer_ckpt else None,
            "has_detector": bool(pipeline and pipeline.has_detector),
            "has_samples": bool(catalog.collections()),
        }
    )


@app.get("/api/samples/collections")
def sample_collections() -> JSONResponse:
    return JSONResponse(
        {
            "collections": [
                {
                    "key": c.key,
                    "label": c.label,
                    "count": len(catalog.images(c.key)),
                    "has_ground_truth": c.manifest_root is not None,
                }
                for c in catalog.collections()
            ]
        }
    )


@app.get("/api/samples")
def list_samples(
    dataset: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(48, ge=1, le=PAGE_SIZE_LIMIT),
    q: str = Query(""),
) -> JSONResponse:
    if catalog.get(dataset) is None:
        raise HTTPException(status_code=404, detail=f"unknown collection: {dataset}")

    names = catalog.images(dataset)
    if q:
        needle = q.lower()
        names = [n for n in names if needle in n.lower()]

    window = names[offset : offset + limit]
    return JSONResponse(
        {
            "dataset": dataset,
            "total": len(names),
            "offset": offset,
            "limit": limit,
            "items": [{"name": name} for name in window],
        }
    )


@app.get("/api/samples/thumb")
def sample_thumb(dataset: str = Query(...), path: str = Query(...)) -> Response:
    resolved = catalog.resolve(dataset, path)
    if resolved is None or not resolved.is_file():
        raise HTTPException(status_code=404, detail="sample not found")
    stat = resolved.stat()
    try:
        payload = _thumbnail_png(str(resolved), stat.st_mtime, stat.st_size)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"could not render thumbnail: {exc}") from exc
    return Response(content=payload, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/ocr")
async def ocr(
    file: UploadFile | None = File(None),
    dataset: str = Form(""),
    path: str = Form(""),
    use_detector: bool = Form(True),
    threshold: float = Form(0.35),
    split_lines: bool = Form(True),
) -> JSONResponse:
    pipeline = store.require()

    ground_truth: str | None = None
    related: list[dict[str, str]] = []
    if dataset and path:
        resolved = catalog.resolve(dataset, path)
        if resolved is None or not resolved.is_file():
            raise HTTPException(status_code=404, detail="sample not found")
        source_name = path
        source = f"{dataset}/{path}"
        ground_truth = catalog.ground_truth_for(dataset, path)
        related = catalog.counterparts(dataset, path)
        try:
            image = Image.open(resolved).convert("L")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail=f"could not read that sample: {exc}") from exc
    elif file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix and suffix not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"unsupported file type: {suffix}")

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

        source_name = file.filename or "upload"
        source = source_name
        try:
            image = Image.open(io.BytesIO(raw)).convert("L")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"could not read that image: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail="send a file, or a dataset and path")

    started = time.perf_counter()
    result = pipeline.read_image(
        image,
        use_detector=use_detector and pipeline.has_detector,
        threshold=max(0.01, min(0.99, threshold)),
        split_lines=split_lines,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    payload = {
        "filename": source_name,
        "source": source,
        "text": result.text,
        "lines": [line.to_dict() for line in result.lines],
        "width": image.width,
        "height": image.height,
        "image": _png_data_uri(image),
        "used_line_split_fallback": result.used_line_split_fallback,
        "used_detector": use_detector and pipeline.has_detector,
        "elapsed_ms": round(elapsed_ms, 1),
        "ground_truth": ground_truth,
        "related": related,
    }
    if ground_truth is not None:
        payload["cer"] = round(compute_cer(result.text, ground_truth), 4)
        payload["wer"] = round(compute_wer(result.text, ground_truth), 4)
    return JSONResponse(payload)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the drag-and-drop OCR web app.")
    parser.add_argument("--recognizer_ckpt", default=os.environ.get("OCR_RECOGNIZER_CKPT", "checkpoints/recognizer_last.pt"))
    parser.add_argument("--detector_ckpt", default=os.environ.get("OCR_DETECTOR_CKPT", "checkpoints/detector_last.pt"))
    parser.add_argument("--no_detector", action="store_true", help="serve recognizer-only (every image is one line)")
    parser.add_argument(
        "--samples_dir",
        action="append",
        default=None,
        help="directory to browse for sample images; repeatable (default: data)",
    )
    parser.add_argument("--no_samples", action="store_true", help="disable the dataset browser")
    parser.add_argument("--device", default=os.environ.get("OCR_DEVICE", "auto"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def anchored(path: str | Path) -> Path:
    """Resolve a relative path against the project root, not the launch directory.

    Without this, `data/` and `checkpoints/` only resolve when the server happens
    to be started from the project root, and the dataset browser silently
    disappears everywhere else.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    from_cwd = Path.cwd() / path
    return from_cwd if from_cwd.exists() else PROJECT_ROOT / path


def sample_roots(args) -> list[Path]:
    if args.no_samples:
        return []
    roots = args.samples_dir or [d for d in os.environ.get("OCR_SAMPLES_DIR", "data").split(os.pathsep) if d]
    return [anchored(root) for root in roots]


def main() -> None:
    import uvicorn

    global catalog
    args = build_arg_parser().parse_args()

    store.load(
        recognizer_ckpt=anchored(args.recognizer_ckpt),
        detector_ckpt=None if args.no_detector else anchored(args.detector_ckpt),
        device=args.device,
    )
    if store.error:
        print(f"warning: {store.error}")
        print("the server will start, but /api/ocr returns 503 until the checkpoints load")
    else:
        print(f"models loaded on {store.pipeline.device}")

    roots = sample_roots(args)
    catalog = DatasetCatalog(roots=roots, base=PROJECT_ROOT)
    collections = catalog.collections()
    if collections:
        print(f"browsable sample collections ({len(collections)}):")
        for c in collections:
            gt = " (with ground truth)" if c.manifest_root else ""
            print(f"  {c.key}{gt}")
    elif roots:
        print(f"no sample collections found under: {', '.join(str(r) for r in roots)}")
        print("the dataset browser will say so in the page; pass --samples_dir to point it elsewhere")
    else:
        print("dataset browser disabled; upload-only mode")

    print(f"open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

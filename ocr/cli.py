"""Command-line entry point. Run as `python main.py --mode ...` or `python -m ocr.cli`."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ocr.config import (
    DEFAULT_CHARSET,
    DEFAULT_DETECTOR_SIZE,
    DEFAULT_PAGE_SIZE,
    DEFAULT_RECOGNIZER_HEIGHT,
    DEFAULT_RECOGNIZER_WIDTH,
    LEGACY_RECOGNIZER_SIZE,
)
from ocr.data import (
    DegradationConfig,
    DetectionDataset,
    DetectionTransform,
    PrintedLineDataset,
    RecognitionTransform,
    detection_collate,
    generate_synthetic_dataset,
    recognition_collate,
)
from ocr.data.targets import DEFAULT_SHRINK_RATIO
from ocr.evaluation import benchmark, evaluate_end_to_end
from ocr.evaluation.compare import (
    compare_systems,
    format_table,
    pipeline_reader,
    tesseract_available,
    tesseract_reader,
    tesseract_version,
)
from ocr.inference import OCRPipeline, run_ocr_folder
from ocr.models import CRNNRecognizer, DBNet
from ocr.training import (
    RunTracker,
    default_run_name,
    load_checkpoint,
    save_checkpoint,
    train_detector,
    train_recognizer,
    validate_detector,
    validate_recognizer,
)
from ocr.utils import resolve_device, set_seed


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Expected WxH, for example 768x1024.")
    w, h = value.lower().split("x", 1)
    return int(w), int(h)


def build_loader(dataset, args, shuffle: bool, collate_fn=None) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def build_detector(args, device: torch.device) -> DBNet:
    model = DBNet().to(device)
    if args.detector_ckpt:
        load_checkpoint(args.detector_ckpt, model, device=device, strict=False)
    return maybe_compile(model, args.compile_model)


def build_recognizer(args, device: torch.device) -> CRNNRecognizer:
    model = CRNNRecognizer(charset=DEFAULT_CHARSET).to(device)
    if args.recognizer_ckpt:
        load_checkpoint(args.recognizer_ckpt, model, device=device, strict=False)
    return maybe_compile(model, args.compile_model)


def detector_ckpt_meta(args) -> dict:
    """shrink/unclip ratios the detector checkpoint was trained with (both 0 for pre-DB checkpoints)."""
    if args.detector_ckpt and not getattr(args, "skip_detector", False):
        ckpt = torch.load(args.detector_ckpt, map_location="cpu", weights_only=False)
        return {
            "shrink_ratio": float(ckpt.get("shrink_ratio", 0.0)),
            "unclip_ratio": float(ckpt.get("unclip_ratio", 0.0)),
            "is_db": ckpt.get("detector_loss") == "db",
        }
    return {"shrink_ratio": 0.0, "unclip_ratio": 0.0, "is_db": False}


def detector_unclip_ratio(args) -> float:
    """--unclip_ratio if given, else whatever the detector checkpoint recorded."""
    if args.unclip_ratio is not None:
        return args.unclip_ratio
    return detector_ckpt_meta(args)["unclip_ratio"]


def recognizer_input_size(args) -> tuple[int, int]:
    """(height, width) the recognizer checkpoint was trained at; explicit flags win; old checkpoints = legacy 32x512."""
    explicit = (args.rec_height, args.rec_width)
    if explicit != (DEFAULT_RECOGNIZER_HEIGHT, DEFAULT_RECOGNIZER_WIDTH):
        return explicit
    if args.recognizer_ckpt:
        ckpt = torch.load(args.recognizer_ckpt, map_location="cpu", weights_only=False)
        if "rec_height" in ckpt:
            return int(ckpt["rec_height"]), int(ckpt["rec_width"])
        return LEGACY_RECOGNIZER_SIZE
    return explicit


def build_pipeline(args, device: torch.device) -> OCRPipeline:
    rec_h, rec_w = recognizer_input_size(args)
    return OCRPipeline(
        recognizer=build_recognizer(args, device),
        detector=None if args.skip_detector else build_detector(args, device),
        device=device,
        charset=DEFAULT_CHARSET,
        detector_size=args.detector_size,
        threshold=args.threshold,
        rec_height=rec_h,
        rec_width=rec_w,
        split_lines_without_detector=args.split_lines_without_detector,
        unclip_ratio=detector_unclip_ratio(args),
        # auto = the deployed default: use the learned threshold whenever the detector has one
        learned_threshold=(detector_ckpt_meta(args)["is_db"] if args.learned_threshold == "auto" else args.learned_threshold == "on"),
    )


def build_scheduler(args, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    """Warmup then cosine decay to 2% of --lr over the whole run, per optimizer step. None = constant."""
    if args.lr_schedule == "none":
        return None
    total = max(1, args.epochs * max(1, steps_per_epoch // max(1, args.grad_accum_steps)))
    warmup = min(args.warmup_steps, total // 10)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    print(f"lr schedule: cosine, {warmup} warmup steps, {total} total steps, floor {0.02 * args.lr:.2e}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def open_tracker(args) -> RunTracker:
    """One run directory per training invocation, under --log_dir."""
    tracker = RunTracker(args.log_dir, args.run_name or default_run_name(args.mode), enabled=not args.no_tensorboard)
    if tracker.enabled:
        tracker.hparams({k: v for k, v in vars(args).items() if v is not None})
        print(f"logging metrics to {tracker.run_dir}")
        if tracker.tensorboard_available:
            print(f"watch live with:  tensorboard --logdir {args.log_dir}")
        else:
            print("tensorboard is not installed; metrics.jsonl is still written (pip install tensorboard)")
    return tracker


def mode_generate(args) -> None:
    cfg = DegradationConfig(
        gaussian_noise=not args.no_gaussian_noise,
        salt_pepper=not args.no_salt_pepper,
        blur=not args.no_blur,
        low_contrast=not args.no_low_contrast,
        uneven_illumination=not args.no_uneven_illumination,
        smudges=not args.no_smudges,
        stains=not args.no_stains,
        faded_print=not args.no_faded_print,
        speckle=not args.no_speckle,
        compression=args.compression,
        severity=args.degradation_severity,
    )
    paths = generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        page_size=args.page_size,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        make_line_crops=not args.no_line_crops,
        line_crop_truncation_prob=args.line_crop_truncation_prob,
        page_crop_prob=args.page_crop_prob,
        degradation=cfg,
        preview_count=args.preview_count,
        seed=args.seed,
        log_every=args.log_every,
        render_profile=args.render_profile,
        val_split=args.val_split,
        workers=args.workers,
    )
    print("synthetic data written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def mode_train_detector(args, device: torch.device) -> None:
    # "db": DBNet objective on shrunk targets; "bce": the original full-box BCE.
    shrink = args.shrink_ratio if args.detector_loss == "db" else 0.0
    unclip = shrink  # the short-side rule inverts exactly with the same ratio
    print(f"detector loss: {args.detector_loss} (shrink_ratio {shrink}, unclip_ratio {unclip})")

    transform = DetectionTransform(size=args.detector_size, augment=args.augment)
    train_ds = DetectionDataset(
        args.detector_manifest, root_dir=args.data_root, image_size=args.detector_size, transform=transform, shrink_ratio=shrink,
    )
    train_loader = build_loader(train_ds, args, shuffle=True, collate_fn=detection_collate)
    val_loader = None
    if args.val_detector_manifest:
        val_ds = DetectionDataset(
            args.val_detector_manifest,
            root_dir=args.data_root,
            image_size=args.detector_size,
            transform=DetectionTransform(size=args.detector_size, augment=False),
            shrink_ratio=shrink,
        )
        val_loader = build_loader(val_ds, args, shuffle=False, collate_fn=detection_collate)

    model = build_detector(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_tracker(args) as tracker:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            train_stats = train_detector(
                model, train_loader, optimizer, device, epoch, args.amp, args.grad_accum_steps, args.log_every,
                tracker=tracker, global_step=global_step, loss_name=args.detector_loss, scheduler=scheduler,
            )
            global_step = train_stats["global_step"]
            print(f"detector epoch {epoch}: train_loss={train_stats['loss']:.4f}")
            if val_loader is not None:
                val_stats = validate_detector(
                    model, val_loader, device, args.amp, 0.5, tracker=tracker, step=global_step,
                    unclip_ratio=unclip, box_threshold=args.threshold,
                )
                print(
                    "detector val: "
                    f"loss={val_stats['loss']:.4f} pixel_f1={val_stats['f1']:.3f} | "
                    f"box_f1={val_stats['box_f1']:.3f} (p={val_stats['box_precision']:.3f} r={val_stats['box_recall']:.3f}) "
                    f"detector_only_box_f1={val_stats['box_f1_detector_only']:.3f}"
                )
            save_checkpoint(
                output_dir / "detector_last.pt", model, optimizer, epoch,
                {"detector_loss": args.detector_loss, "shrink_ratio": shrink, "unclip_ratio": unclip},
            )


def mode_train_recognizer(args, device: torch.device) -> None:
    train_ds = PrintedLineDataset(
        args.line_manifest,
        root_dir=args.data_root,
        charset=DEFAULT_CHARSET,
        transform=RecognitionTransform(args.rec_height, args.rec_width, augment=args.augment),
    )
    train_loader = build_loader(train_ds, args, shuffle=True, collate_fn=recognition_collate)
    val_loader = None
    if args.val_line_manifest:
        val_ds = PrintedLineDataset(
            args.val_line_manifest,
            root_dir=args.data_root,
            charset=DEFAULT_CHARSET,
            transform=RecognitionTransform(args.rec_height, args.rec_width, augment=False),
        )
        val_loader = build_loader(val_ds, args, shuffle=False, collate_fn=recognition_collate)

    model = build_recognizer(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_tracker(args) as tracker:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            train_stats = train_recognizer(
                model, train_loader, optimizer, device, epoch, args.amp, args.grad_accum_steps, args.log_every,
                tracker=tracker, global_step=global_step, scheduler=scheduler,
            )
            global_step = train_stats["global_step"]
            print(f"recognizer epoch {epoch}: train_loss={train_stats['loss']:.4f}")
            if val_loader is not None:
                val_stats = validate_recognizer(model, val_loader, device, DEFAULT_CHARSET, args.amp, tracker=tracker, step=global_step)
                print(
                    "recognizer val: "
                    f"loss={val_stats['loss']:.4f} cer={val_stats['cer']:.3f} "
                    f"wer={val_stats['wer']:.3f} exact={val_stats['exact_match']:.3f}"
                )
            save_checkpoint(
                output_dir / "recognizer_last.pt", model, optimizer, epoch,
                {"charset": DEFAULT_CHARSET, "rec_height": args.rec_height, "rec_width": args.rec_width},
            )


def mode_test_detector(args, device: torch.device) -> None:
    meta = detector_ckpt_meta(args)  # score pixels against the targets this checkpoint was trained on
    ds = DetectionDataset(
        args.detector_manifest,
        root_dir=args.data_root,
        image_size=args.detector_size,
        transform=DetectionTransform(size=args.detector_size, augment=False),
        shrink_ratio=meta["shrink_ratio"],
    )
    loader = build_loader(ds, args, shuffle=False, collate_fn=detection_collate)
    model = build_detector(args, device)
    stats = validate_detector(model, loader, device, args.amp, 0.5, unclip_ratio=detector_unclip_ratio(args), box_threshold=args.threshold)
    print(json.dumps(stats, indent=2))


def mode_test_recognizer(args, device: torch.device) -> None:
    ds = PrintedLineDataset(
        args.line_manifest,
        root_dir=args.data_root,
        charset=DEFAULT_CHARSET,
        transform=RecognitionTransform(args.rec_height, args.rec_width, augment=False),
    )
    loader = build_loader(ds, args, shuffle=False, collate_fn=recognition_collate)
    model = build_recognizer(args, device)
    stats = validate_recognizer(model, loader, device, DEFAULT_CHARSET, args.amp)
    print(json.dumps(stats, indent=2))


def mode_ocr_folder(args, device: torch.device) -> None:
    if args.skip_detector and args.split_lines_without_detector:
        print("ocr mode: detector skipped; splitting each page into line crops by image projection")
    elif args.skip_detector:
        print("ocr mode: detector skipped; each image is one recognizer crop")
    else:
        print("ocr mode: detector enabled with line-splitting fallback")

    results = run_ocr_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pipeline=build_pipeline(args, device),
        save_crops=args.save_crops,
        save_visualizations=args.save_visualizations,
        log_every=args.log_every,
    )
    print(f"wrote OCR for {len(results)} image(s) to {args.output_dir}")


def mode_benchmark(args, device: torch.device) -> None:
    predictions_path = Path(args.predictions_json) if args.predictions_json else Path(args.output_dir) / "predictions.json"
    if not predictions_path.exists():
        if not args.input_dir:
            raise ValueError("Provide --predictions_json or --input_dir for benchmark mode.")
        mode_ocr_folder(args, device)
    report = benchmark(predictions_path, labels_path=args.labels_manifest, detection_labels_path=args.detector_labels_manifest)
    print(json.dumps(report, indent=2))


def mode_eval_e2e(args, device: torch.device) -> None:
    """Full pipeline on held-out pages -> corpus CER/WER + box F1, written to --output_dir and the tracker."""
    pipeline = build_pipeline(args, device)
    print(f"e2e eval: {args.detector_manifest} (unclip_ratio {pipeline.unclip_ratio})")
    report = evaluate_end_to_end(
        pipeline, args.detector_manifest, root_dir=args.data_root, limit=args.limit or None, log_every=args.log_every,
    )
    summary = report.pop("breakdown_summary")
    print(json.dumps({k: v for k, v in report.items() if k != "breakdown"}, indent=2))
    print("--- error breakdown ---")
    print(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e2e_report.json").write_text(json.dumps(report, indent=2))
    with open_tracker(args) as tracker:
        if tracker.enabled:
            tracker.scalars({k: v for k, v in report.items() if isinstance(v, (int, float))}, 0, prefix="e2e/")
    print(
        f"E2E  cer {report['cer']:.4f}  wer {report['wer']:.4f}  page_exact {report['page_exact']:.3f}  "
        f"det_f1 {report['det_f1']:.3f}  ({report['num_pages']} pages) -> {output_dir / 'e2e_report.json'}"
    )


def mode_pseudo_label(args) -> None:
    """Label real pages from their clean twins with Tesseract (see ocr/data/pseudo_label.py)."""
    from ocr.data.pseudo_label import label_pairs

    label_pairs(Path(args.clean_dir), Path(args.input_dir), Path(args.output_dir), min_conf=args.min_conf)


def mode_compare(args, device: torch.device) -> None:
    """This pipeline against baselines and other checkpoints on the same pages."""
    readers: list[tuple[str, object]] = []

    if not args.no_self:
        readers.append((args.label or "this pipeline", pipeline_reader(build_pipeline(args, device))))

    # --compare_model "name=recognizer.pt:detector.pt" - repeatable, for older runs
    for spec in args.compare_model or []:
        name, _, paths = spec.partition("=")
        rec, _, det = paths.partition(":")
        if not name or not rec:
            raise ValueError(f'--compare_model expects "name=recognizer.pt[:detector.pt]", got {spec!r}')
        for label, path in (("recognizer", rec), ("detector", det)):
            if path and not Path(path).is_file():
                raise FileNotFoundError(f"--compare_model {name!r}: {label} checkpoint not found: {path}")
        other = OCRPipeline.load(rec, det or None, device=device, charset=DEFAULT_CHARSET, detector_size=args.detector_size, threshold=args.threshold)
        readers.append((name, pipeline_reader(other)))

    for baseline in (b.strip() for b in args.baselines.split(",") if b.strip()):
        if baseline != "tesseract":
            raise ValueError(f"unknown baseline {baseline!r} (available: tesseract)")
        if not tesseract_available():
            print("skipping tesseract baseline: not installed (brew install tesseract / apt install tesseract-ocr)")
            continue
        readers.append((tesseract_version(), tesseract_reader(psm=args.tesseract_psm)))

    if not readers:
        raise ValueError("nothing to compare: pass checkpoints, --compare_model, or --baselines tesseract")

    print(f"comparing {len(readers)} system(s) on {args.detector_manifest}")
    rows = compare_systems(args.detector_manifest, readers, root_dir=args.data_root, limit=args.limit or None, log_every=args.log_every)
    table = format_table(rows)
    print()
    print(table)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps({"results": rows}, indent=2))
    (output_dir / "comparison.txt").write_text(table + "\n")
    print(f"\nwritten to {output_dir / 'comparison.json'}")


def mode_smoke_test(args, device: torch.device) -> None:
    smoke_dir = Path(args.output_dir) / "smoke_synth"
    generate_synthetic_dataset(
        smoke_dir,
        num_samples=2,
        page_size=args.page_size,
        preview_count=2,
        seed=args.seed,
        log_every=1,
        render_profile=args.render_profile,
    )

    detector = DBNet().to(device)
    recognizer = CRNNRecognizer().to(device)
    x_page = torch.randn(1, 1, args.detector_size[1], args.detector_size[0], device=device)
    x_line = torch.randn(1, 1, args.rec_height, args.rec_width, device=device)
    with torch.no_grad():
        det_out = detector(x_page)
        rec_out = recognizer(x_line)
    print(f"detector forward: {tuple(det_out.shape)}")
    print(f"recognizer forward: {tuple(rec_out.shape)}")
    print(f"synthetic previews: {smoke_dir / 'previews'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Practical OCR for degraded printed document text.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "generate_synth_data",
            "train_detector",
            "train_recognizer",
            "test_detector",
            "test_recognizer",
            "benchmark",
            "ocr_folder",
            "eval_e2e",
            "pseudo_label",
            "compare",
            "smoke_test",
        ],
    )
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--detector_manifest", type=str)
    parser.add_argument("--val_detector_manifest", type=str)
    parser.add_argument("--line_manifest", type=str)
    parser.add_argument("--val_line_manifest", type=str)
    parser.add_argument("--labels_manifest", type=str)
    parser.add_argument("--detector_labels_manifest", type=str)
    parser.add_argument("--predictions_json", type=str)
    parser.add_argument("--detector_ckpt", type=str)
    parser.add_argument("--recognizer_ckpt", type=str)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule", choices=["cosine", "none"], default="cosine", help="cosine: warmup then decay to 2%% of --lr; none: constant")
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard/metrics root; one subdir per run")
    parser.add_argument("--run_name", type=str, help="subdir under --log_dir (default: <mode>_<timestamp>)")
    parser.add_argument("--no_tensorboard", action="store_true", help="disable metrics logging for this run")

    parser.add_argument("--detector_size", type=parse_size, default=DEFAULT_DETECTOR_SIZE)
    parser.add_argument("--rec_height", type=int, default=DEFAULT_RECOGNIZER_HEIGHT)
    parser.add_argument("--rec_width", type=int, default=DEFAULT_RECOGNIZER_WIDTH)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--detector_loss", choices=["db", "bce"], default="db", help="db: DBNet objective on shrunk targets (default); bce: original full-box BCE")
    parser.add_argument("--shrink_ratio", type=float, default=DEFAULT_SHRINK_RATIO, help="fraction of a box's short side removed from each edge for db targets")
    parser.add_argument("--unclip_ratio", type=float, help="override how far detected boxes are re-expanded (default: read from the detector checkpoint)")
    parser.add_argument("--limit", type=int, default=0, help="eval_e2e: only the first N pages")
    parser.add_argument("--clean_dir", type=str, help="pseudo_label: directory of clean twins of --input_dir pages")
    parser.add_argument("--min_conf", type=float, default=60.0, help="pseudo_label: drop Tesseract lines below this mean word confidence")
    parser.add_argument("--baselines", type=str, default="tesseract", help="compare: comma list of external engines to score alongside (tesseract, or empty for none)")
    parser.add_argument("--compare_model", action="append", help='compare: extra checkpoints as "name=recognizer.pt[:detector.pt]"; repeatable')
    parser.add_argument("--label", type=str, help="compare: name for the current checkpoints in the table")
    parser.add_argument("--no_self", action="store_true", help="compare: leave the current checkpoints out of the table")
    parser.add_argument("--tesseract_psm", type=int, default=6, help="compare: Tesseract page segmentation mode (6 = uniform block of text)")
    parser.add_argument("--learned_threshold", choices=["auto", "on", "off"], default="auto",
                        help="binarize with the detector's learned threshold map (auto: on for DB-trained checkpoints)")
    parser.add_argument("--skip_detector", action="store_true")
    parser.add_argument("--split_lines_without_detector", action="store_true")
    parser.add_argument("--save_crops", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")

    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1, help="generate_synth_data: parallel renderer processes (page-seeded, so output is identical for any count)")
    parser.add_argument("--page_size", type=parse_size, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--min_lines", type=int, default=7)
    parser.add_argument("--max_lines", type=int, default=10)
    parser.add_argument("--preview_count", type=int, default=12)
    parser.add_argument("--no_line_crops", action="store_true")
    parser.add_argument("--line_crop_truncation_prob", type=float, default=0.20)
    parser.add_argument("--page_crop_prob", type=float, default=0.35)
    parser.add_argument("--render_profile", choices=["noisyoffice", "document"], default="noisyoffice")
    parser.add_argument(
        "--val_split", type=float, default=0.0,
        help="fraction of pages held out; writes *_manifest_train.json and *_manifest_val.json (split by page)",
    )
    parser.add_argument("--degradation_severity", type=float, default=0.7)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--no_gaussian_noise", action="store_true")
    parser.add_argument("--no_salt_pepper", action="store_true")
    parser.add_argument("--no_blur", action="store_true")
    parser.add_argument("--no_low_contrast", action="store_true")
    parser.add_argument("--no_uneven_illumination", action="store_true")
    parser.add_argument("--no_smudges", action="store_true")
    parser.add_argument("--no_stains", action="store_true")
    parser.add_argument("--no_faded_print", action="store_true")
    parser.add_argument("--no_speckle", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.mode == "generate_synth_data":
        mode_generate(args)
    elif args.mode == "train_detector":
        if not args.detector_manifest:
            raise ValueError("--detector_manifest is required for train_detector")
        mode_train_detector(args, device)
    elif args.mode == "train_recognizer":
        if not args.line_manifest:
            raise ValueError("--line_manifest is required for train_recognizer")
        mode_train_recognizer(args, device)
    elif args.mode == "test_detector":
        if not args.detector_manifest or not args.detector_ckpt:
            raise ValueError("--detector_manifest and --detector_ckpt are required for test_detector")
        mode_test_detector(args, device)
    elif args.mode == "test_recognizer":
        if not args.line_manifest or not args.recognizer_ckpt:
            raise ValueError("--line_manifest and --recognizer_ckpt are required for test_recognizer")
        mode_test_recognizer(args, device)
    elif args.mode == "ocr_folder":
        if not args.input_dir or not args.recognizer_ckpt:
            raise ValueError("--input_dir and --recognizer_ckpt are required for ocr_folder")
        if args.split_lines_without_detector and not args.skip_detector:
            raise ValueError("--split_lines_without_detector is only used together with --skip_detector")
        if not args.skip_detector and not args.detector_ckpt:
            raise ValueError("--detector_ckpt is required for full-page OCR. Use --skip_detector for cropped inputs.")
        mode_ocr_folder(args, device)
    elif args.mode == "benchmark":
        mode_benchmark(args, device)
    elif args.mode == "pseudo_label":
        if not args.input_dir or not args.clean_dir:
            raise ValueError("--input_dir (noisy pages) and --clean_dir (clean twins) are required for pseudo_label")
        mode_pseudo_label(args)
    elif args.mode == "compare":
        if not args.detector_manifest:
            raise ValueError("--detector_manifest (pages with line text) is required for compare")
        if not args.no_self and not args.recognizer_ckpt:
            raise ValueError("--recognizer_ckpt is required for compare (or pass --no_self to compare only baselines)")
        mode_compare(args, device)
    elif args.mode == "eval_e2e":
        if not args.detector_manifest or not args.recognizer_ckpt:
            raise ValueError("--detector_manifest (pages with line text) and --recognizer_ckpt are required for eval_e2e")
        mode_eval_e2e(args, device)
    elif args.mode == "smoke_test":
        mode_smoke_test(args, device)


if __name__ == "__main__":
    main()

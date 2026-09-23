"""Train/validate loops for the detector and the recognizer.

Each loop takes an optional RunTracker. Training loops log at the same points
they print, keyed by a global step that carries across epochs so the curves in
TensorBoard are continuous; validation logs once per call at that same step,
plus a look at actual predictions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ocr.config import DEFAULT_CHARSET
from ocr.ctc import decode_ctc_greedy
from PIL import Image

from ocr.evaluation.metrics import compute_cer, compute_wer, detection_counts, exact_match_ratio, prf_from_counts
from ocr.inference.detection import boxes_from_prob_map
from ocr.training.losses import DBLoss, bce_loss
from ocr.training.tracking import RunTracker
from ocr.utils import ProgressLogger

SAMPLE_PREDICTIONS = 12  # rows in the TensorBoard text table
SAMPLE_MASKS = 4  # detector images shown per validation


def _amp_enabled(device: torch.device, amp: bool) -> bool:
    return amp and device.type == "cuda"


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, enabled=_amp_enabled(device, amp))


def _grad_scaler(device: torch.device, amp: bool):
    enabled = _amp_enabled(device, amp)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _prediction_table(predictions: list[str], targets: list[str]) -> str:
    rows = ["| # | target | prediction | CER |", "| --- | --- | --- | --- |"]
    for i, (pred, target) in enumerate(zip(predictions, targets), start=1):
        mark = "✓" if pred == target else ""
        rows.append(f"| {i} | `{_escape_md(target)}` | `{_escape_md(pred)}` {mark} | {compute_cer(pred, target):.2f} |")
    return "\n".join(rows)


def _to_pil(image: torch.Tensor) -> Image.Image:
    """[1,H,W] normalized tensor -> grayscale PIL, for the projection-based line splitter."""
    arr = ((image[0].float() * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr, mode="L")


def _detector_panels(images: torch.Tensor, gt: torch.Tensor, probs: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    """[N,1,H,W] x4 -> [N,1,H,4W]: input | target | probability | threshold, side by side."""
    inputs = (images.float() * 0.5 + 0.5).clamp(0, 1)
    return torch.cat([inputs, gt.float(), probs.float(), thresh.float()], dim=-1).cpu()


def train_detector(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    tracker: RunTracker | None = None,
    global_step: int = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    loss_name: str = "db",
) -> dict[str, float]:
    """One epoch. loss_name: "db" (DBNet objective, default) or "bce" (the original)."""
    model.train()
    db_loss = DBLoss()
    scaler = _grad_scaler(device, amp)
    totals: dict[str, float] = {}
    progress = ProgressLogger(f"detector train epoch {epoch}", len(train_loader), log_every)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        global_step += 1
        images = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        with _autocast(device, amp):
            if loss_name == "db":
                prob_logits, thresh_logits = model.forward_maps(images)
                terms = db_loss(
                    prob_logits, thresh_logits, gt,
                    batch["thresh_map"].to(device, non_blocking=True),
                    batch["thresh_mask"].to(device, non_blocking=True),
                )
            else:
                prob_logits = model(images)
                terms = bce_loss(prob_logits.float(), gt)
            loss = terms["loss"] / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value)

        if log_every and progress.should_log(step):
            with torch.no_grad():
                preds = torch.sigmoid(prob_logits.detach()) >= 0.5
                truth = gt >= 0.5
                precision, recall, f1 = _prf(
                    float((preds & truth).sum()),
                    float((preds & ~truth).sum()),
                    float((~preds & truth).sum()),
                )
            means = {name: value / step for name, value in totals.items()}
            extra = " ".join(f"{k[5:]} {v:.3f}" for k, v in means.items() if k != "loss")
            progress.log(step, f"loss {means['loss']:.4f} [{extra}] precision {precision:.3f} recall {recall:.3f} f1 {f1:.3f}")
            if tracker:
                tracker.scalars(
                    {**means, "precision": precision, "recall": recall, "f1": f1, "lr": _lr(optimizer)},
                    global_step,
                    prefix="train/",
                )

    if len(train_loader) % grad_accum_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    epoch_loss = totals.get("loss", 0.0) / max(1, len(train_loader))
    if tracker:
        tracker.scalar("epoch/train_loss", epoch_loss, epoch)
        tracker.flush()
    return {"loss": epoch_loss, "global_step": global_step}


@torch.no_grad()
def validate_detector(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp: bool = False,
    threshold: float = 0.5,
    tracker: RunTracker | None = None,
    step: int = 0,
    unclip_ratio: float = 0.0,
    box_threshold: float = 0.35,
) -> dict[str, float]:
    """Pixel P/R/F1 against the target mask AND box-level P/R/F1 at IoU >= 0.5.

    The box metric runs the real postprocessing (`boxes_from_prob_map`, the same
    call inference makes) on each page and matches against the manifest boxes,
    so it reflects whether lines come out separated - which pixel F1 cannot see.
    `box_threshold` is the inference-time cutoff; `threshold` is for pixel PRF.
    """
    model.eval()
    losses: list[float] = []
    pixel_tp = pixel_fp = pixel_fn = 0.0
    box_tp = box_fp = box_fn = 0
    raw_tp = raw_fp = raw_fn = 0  # detector alone, before the projection-split rescue
    progress = ProgressLogger("detector val", len(val_loader), log_every=20)
    panels: torch.Tensor | None = None

    for batch_idx, batch in enumerate(val_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        with _autocast(device, amp):
            prob_logits, thresh_logits = model.forward_maps(images)
            loss = F.binary_cross_entropy_with_logits(prob_logits.float(), gt)
        probs = torch.sigmoid(prob_logits.float())
        preds = probs >= threshold
        truth = gt >= 0.5
        pixel_tp += float((preds & truth).sum())
        pixel_fp += float((preds & ~truth).sum())
        pixel_fn += float((~preds & truth).sum())
        losses.append(float(loss))

        # Box-level: the postprocessing OCR will actually run, page by page.
        # Score the path inference actually takes: for a DB detector that is the
        # learned threshold (B = sigmoid(50 (P - T)) cut at 0.5) plus the T-ridge unclip.
        thresh_probs = torch.sigmoid(thresh_logits.float())
        if unclip_ratio > 0:
            score_maps = torch.sigmoid(50.0 * (probs - thresh_probs))
            cutoff = 0.5
        else:
            score_maps, cutoff = probs, box_threshold
        for i in range(images.shape[0]):
            pil, gt_boxes = _to_pil(images[i]), [tuple(b) for b in batch["boxes"][i]]
            score_np = score_maps[i, 0].cpu().numpy()
            t_np = thresh_probs[i, 0].cpu().numpy() if unclip_ratio > 0 else None
            tp, fp, fn = detection_counts(boxes_from_prob_map(score_np, pil, cutoff, unclip_ratio, thresh_map=t_np), gt_boxes)
            box_tp, box_fp, box_fn = box_tp + tp, box_fp + fp, box_fn + fn
            tp, fp, fn = detection_counts(boxes_from_prob_map(score_np, pil, cutoff, unclip_ratio, split_lines=False, thresh_map=t_np), gt_boxes)
            raw_tp, raw_fp, raw_fn = raw_tp + tp, raw_fp + fp, raw_fn + fn

        if panels is None and tracker:
            k = min(SAMPLE_MASKS, images.shape[0])
            panels = _detector_panels(images[:k], gt[:k], probs[:k], torch.sigmoid(thresh_logits[:k].float()))
        if progress.should_log(batch_idx):
            precision, recall, f1 = _prf(pixel_tp, pixel_fp, pixel_fn)
            box = prf_from_counts(box_tp, box_fp, box_fn)
            raw = prf_from_counts(raw_tp, raw_fp, raw_fn)
            progress.log(
                batch_idx,
                f"loss {float(np.mean(losses)):.4f} pixel f1 {f1:.3f} | box f1 {box['f1']:.3f} "
                f"(p {box['precision']:.3f} r {box['recall']:.3f}) | detector-only box f1 {raw['f1']:.3f}",
            )

    precision, recall, f1 = _prf(pixel_tp, pixel_fp, pixel_fn)
    box = prf_from_counts(box_tp, box_fp, box_fn)
    raw = prf_from_counts(raw_tp, raw_fp, raw_fn)
    stats = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "box_precision": box["precision"],
        "box_recall": box["recall"],
        "box_f1": box["f1"],
        "box_f1_detector_only": raw["f1"],
    }
    if tracker:
        tracker.scalars(stats, step, prefix="val/")
        if panels is not None:
            tracker.image_grid("val/input_target_prob_thresh", panels, step)
        tracker.flush()
    return stats


def train_recognizer(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    tracker: RunTracker | None = None,
    global_step: int = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = _grad_scaler(device, amp)
    total_loss = 0.0
    progress = ProgressLogger(f"recognizer train epoch {epoch}", len(train_loader), log_every)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        global_step += 1
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full(
                (images.size(0),),
                logits.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total_loss += float(loss.detach()) * grad_accum_steps

        if log_every and progress.should_log(step):
            with torch.no_grad():
                batch_predictions = decode_ctc_greedy(logits.detach(), time_first=True)
            batch_targets = batch["texts"]
            cer = float(np.mean([compute_cer(p, t) for p, t in zip(batch_predictions, batch_targets)])) if batch_targets else 0.0
            exact = exact_match_ratio(batch_predictions, batch_targets) if batch_targets else 0.0
            progress.log(step, f"loss {total_loss / step:.4f} cer {cer:.3f} exact {exact:.3f}")
            if tracker:
                tracker.scalars(
                    {"loss": total_loss / step, "cer": cer, "exact_match": exact, "lr": _lr(optimizer)},
                    global_step,
                    prefix="train/",
                )

    if len(train_loader) % grad_accum_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    epoch_loss = total_loss / max(1, len(train_loader))
    if tracker:
        tracker.scalar("epoch/train_loss", epoch_loss, epoch)
        tracker.flush()
    return {"loss": epoch_loss, "global_step": global_step}


@torch.no_grad()
def validate_recognizer(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    charset: str = DEFAULT_CHARSET,
    amp: bool = False,
    tracker: RunTracker | None = None,
    step: int = 0,
) -> dict[str, float]:
    model.eval()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    losses: list[float] = []
    predictions: list[str] = []
    targets_text: list[str] = []
    progress = ProgressLogger("recognizer val", len(val_loader), log_every=20)

    for batch_idx, batch in enumerate(val_loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        losses.append(float(loss))
        predictions.extend(decode_ctc_greedy(logits, charset=charset, time_first=True))
        targets_text.extend(batch["texts"])
        if progress.should_log(batch_idx):
            cer = float(np.mean([compute_cer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
            wer = float(np.mean([compute_wer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
            exact = exact_match_ratio(predictions, targets_text)
            progress.log(batch_idx, f"loss {float(np.mean(losses)):.4f} cer {cer:.3f} wer {wer:.3f} exact {exact:.3f}")

    cer = float(np.mean([compute_cer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
    wer = float(np.mean([compute_wer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
    stats = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match_ratio(predictions, targets_text),
    }
    if tracker:
        tracker.scalars(stats, step, prefix="val/")
        if predictions:
            tracker.text("val/sample_predictions", _prediction_table(predictions[:SAMPLE_PREDICTIONS], targets_text[:SAMPLE_PREDICTIONS]), step)
        tracker.flush()
    return stats

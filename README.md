# Degraded Printed Document OCR

Small PyTorch OCR project for degraded printed document images.

```text
full page PNG -> lightweight DBNet-style text segmentation -> sorted line crops -> CRNN + CTC -> literal OCR text
cropped PNG   -> CRNN + CTC directly
```

There is no spell correction, lexicon correction, or language-model correction in the default path. If a crop visually contains a partial word, the recognizer output is kept literal.

## Layout

```text
main.py                     CLI entry point (thin wrapper around ocr/cli.py)
ocr/                        the library
  config.py                 charset, image sizes, font lookup tables
  ctc.py                    charset encoding + greedy CTC decode
  utils.py                  device resolution, seeding, progress logging
  cli.py                    argparse front end for every --mode
  models/
    blocks.py               shared conv building blocks
    detector.py             DBNet text segmentation
    recognizer.py           CRNN + CTC head
  data/
    transforms.py           resize/normalize, augmentation, degradation helpers
    datasets.py             recognition, detection, and folder datasets
    synthetic.py            synthetic office-document page and line generation
    catalog.py              browsable sample collections + manifest ground truth
  inference/
    detection.py            probability map -> merged, sorted, split line boxes
    recognition.py          batched crop -> text
    pipeline.py             OCRPipeline + run_ocr_image / run_ocr_folder
  training/
    loops.py                train/validate loops for both models
    checkpoints.py          save/load checkpoints
    tracking.py             TensorBoard + metrics.jsonl run tracker
  evaluation/
    metrics.py              CER, WER, exact match, box IoU, detection F1
    benchmark.py            score predictions.json against label manifests
web/
  server.py                 FastAPI app: drop an image, get the text
  static/index.html         the drop-zone UI
remote/
  recognizer.sky.yaml       SkyPilot launch file: train the recognizer on a RunPod RTX 4090
  detector.sky.yaml         same for the detector
  fetch_results.sh          rsync checkpoints/ and runs/ back down
checkpoints/                trained weights (gitignored)
runs/                       one TensorBoard run per training invocation (gitignored)
data/                       SimulatedNoisyOffice + generated synthetic datasets
outputs/                    OCR results, previews, smoke-test artifacts (gitignored)
```

## Install

```bash
pip install -r requirements.txt
```

CPU works. CUDA is used when available with `--device auto`, or explicitly with `--device cuda`.

## Web App

Drop an image in the browser and get the predicted text back:

```bash
python -m web.server --recognizer_ckpt checkpoints/recognizer_last.pt --detector_ckpt checkpoints/detector_last.pt
```

Then open <http://127.0.0.1:8000>. Drag a file onto the page, click to browse, or paste an image from the clipboard.

The page shows the detected line boxes over your image next to the per-line transcript; hovering either side highlights the matching line. Three controls re-run the current image without reloading the models:

- **Detect lines** — turn off when the image is already a single cropped line.
- **Threshold** — the detector probability cutoff (default `0.35`). Lower finds more text and more noise.
- **Split lines by projection** — split a region into lines by horizontal ink projection when the detector returns one blob.

### Browsing the datasets

Below the drop zone is a **Dataset samples** browser over every image collection found under `--samples_dir` (default `data`). Pick a collection, filter by filename, page through the thumbnails, then **click a thumbnail to read it** or **drag one onto the drop zone**. Nothing is uploaded — the server reads the file from disk by path.

`data/` and `checkpoints/` are resolved relative to the project root, so the browser appears no matter which directory you launch from. If no collections turn up, the page says so and tells you which flag to pass rather than quietly hiding the section.

### Comparing versions of the same page

When a dataset ships several versions of one page, they are linked. NoisyOffice names its files `FontLre_Noisec_TE.png` / `FontLre_Clean_TE.png`, so dropping the version token leaves a key shared by every version, and a **Same page** strip appears under the image with the alternatives — each labelled by its variant (`Noisec`, `Noisef`, `Clean`, …) and its collection. Click one to run it, or drag it onto the drop zone.

Flipping between a noisy scan and its clean twin is the quickest read on how much of the error is degradation rather than the recognizer: on `FontLre_TE` the noisy scan collapses to a single unusable line, while the clean original yields 12 lines of roughly-shaped text.

Where the collection's dataset ships a manifest (`*manifest*.json` beside it, as `data/synthetic_docs` does), the expected transcript comes back with the prediction and the header shows live **CER / WER** for that image. Collections with transcripts are marked `✓ ground truth` in the dropdown. NoisyOffice has no transcripts, so those samples show a prediction only.

This makes the training-distribution gap easy to see directly: a `synthetic_docs/line_crops` sample typically scores 0% CER, a `synthetic_docs/pages` sample lands around 7-14% CER, and a NoisyOffice scan is much worse — the recognizer has only ever seen the synthetic renderer's output.

Directory listings and manifests are read lazily and cached, so a collection with 40k line crops costs nothing until you open it. A sample is only served if its name appears in that collection's listing, which is what blocks path traversal.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--no_detector` | serve recognizer-only; every image is treated as one line |
| `--samples_dir` | directory to browse for samples; repeatable (default `data`) |
| `--no_samples` | disable the dataset browser, upload-only |
| `--device` | `auto` (default), `cpu`, `cuda`, `mps` |
| `--host` / `--port` | bind address, default `127.0.0.1:8000` |

`OCR_RECOGNIZER_CKPT`, `OCR_DETECTOR_CKPT`, `OCR_DEVICE`, and `OCR_SAMPLES_DIR` set the same defaults from the environment. The server binds to localhost; uploads are read into memory, capped at 25 MB, and never written to disk.

The same pipeline is importable:

```python
from PIL import Image
from ocr import OCRPipeline

pipeline = OCRPipeline.load(
    recognizer_ckpt="checkpoints/recognizer_last.pt",
    detector_ckpt="checkpoints/detector_last.pt",
)
result = pipeline.read_image(Image.open("page.png"))
print(result.text)
for line in result.lines:
    print(line.box, line.text)
```

## Generate Synthetic Data

```bash
python main.py \
  --mode generate_synth_data \
  --output_dir data/synthetic_docs \
  --num_samples 5000
```

This creates:

- `data/synthetic_docs/pages/*.png`
- `data/synthetic_docs/line_crops/*.png`
- `data/synthetic_docs/pages_manifest.json` for detector training
- `data/synthetic_docs/lines_manifest.json` for recognizer training
- `data/synthetic_docs/previews/*_boxes.png` for quick inspection

Add `--val_split 0.05` to also write `pages_manifest_{train,val}.json` and `lines_manifest_{train,val}.json`. The split is by **page**: every line crop of a held-out page is held out with it, so the recognizer is never scored on text it trained on. Feed the `_train` file to `--line_manifest` / `--detector_manifest` and the `_val` file to the matching `--val_*` flag.

Synthetic pages default to a NoisyOffice-like `540x258` dense paragraph patch: usually 8-10 mostly upright printed lines, tight spacing, constrained office-style fonts, PNG output, gray paper texture, edge clipping, and configurable degradations.

Useful variants:

```bash
python main.py --mode generate_synth_data --output_dir data/synthetic_docs_small --num_samples 50 --preview_count 20
python main.py --mode generate_synth_data --output_dir data/synthetic_clean --num_samples 1000 --no_stains --no_smudges --degradation_severity 0.3
python main.py --mode generate_synth_data --output_dir data/synthetic_full_pages --render_profile document --page_size 768x1024 --max_lines 14
```

## Train Detector

```bash
python main.py \
  --mode train_detector \
  --detector_manifest data/synthetic_docs/pages_manifest.json \
  --data_root data/synthetic_docs \
  --output_dir checkpoints \
  --batch_size 4 \
  --epochs 20 \
  --lr 3e-4
```

For GPU:

```bash
python main.py --mode train_detector --detector_manifest data/synthetic_docs/pages_manifest.json --data_root data/synthetic_docs --output_dir checkpoints --device cuda --amp --pin_memory --num_workers 4
```

## Train Recognizer

```bash
python main.py \
  --mode train_recognizer \
  --line_manifest data/synthetic_docs/lines_manifest.json \
  --data_root data/synthetic_docs \
  --output_dir checkpoints \
  --batch_size 32 \
  --epochs 30 \
  --lr 3e-4
```

The recognizer keeps uppercase, lowercase, punctuation, digits, and spaces. It trains with CTC and greedy CTC decoding.

## Watching Training Live

Every `train_detector` / `train_recognizer` run writes to `runs/<mode>_<timestamp>/` — TensorBoard event files plus a `metrics.jsonl` mirror and `hparams.json`. Start TensorBoard on the parent directory and it picks up new runs and new points as they arrive:

```bash
tensorboard --logdir runs
```

Then open <http://localhost:6006>. What you get per run:

| Tab | Detector | Recognizer |
| --- | --- | --- |
| **Scalars** `train/` | loss, precision, recall, f1, lr — at every `--log_every` step | loss, cer, exact_match, lr |
| **Scalars** `val/` | loss, precision, recall, f1 — once per epoch | loss, cer, wer, exact_match |
| **Scalars** `epoch/` | train_loss per epoch | train_loss per epoch |
| **Images** | `val/input_target_prediction` — page, target line mask, and the predicted probability map side by side, stacked for the first 4 validation images | — |
| **Text** | `hparams` table | `hparams` table, and `val/sample_predictions`: 12 target/prediction pairs with per-row CER, refreshed each epoch |

The recognizer prediction table is the one to watch. CTC outputs blanks for the first few hundred steps, then fragments, then words; the CER curve alone hides which of those you are in.

Train and validation curves share one global step (batches seen), so the validation points sit on the training curve at the moment they were measured, not on a separate epoch axis.

Flags:

| Flag | Meaning |
| --- | --- |
| `--log_dir` | root for runs (default `runs`) |
| `--run_name` | subdirectory name (default `<mode>_<timestamp>`); reuse it to append to a run |
| `--no_tensorboard` | disable tracking for this run |

If `tensorboard` is not installed, training still writes `metrics.jsonl` — one JSON object per logged point — and tells you so.

## Training Remotely (RunPod via SkyPilot)

One command from your laptop provisions a RunPod GPU, syncs the code, generates the data, trains, and streams the log back. [SkyPilot](https://docs.skypilot.co) does the pod lifecycle; the launch files in `remote/` say what to run.

One-time setup:

```bash
pip install "skypilot[runpod]"
```

Put your RunPod API key where SkyPilot reads it — `~/.runpod/config.toml` with `api_key = "..."`, or `export RUNPOD_API_KEY=...` in your shell. It never goes in the repo. Then `sky check runpod` should report the cloud enabled.

Train:

```bash
sky launch -c ocr remote/recognizer.sky.yaml
sky launch -c ocr remote/detector.sky.yaml     # same cluster name: reuses the box and the generated data
```

Both files target an **RTX 4090** and generate 20k synthetic pages with a 5% page-level validation split on first launch (the setup step is idempotent, so the second launch skips it). Data is generated on the GPU box rather than synced up because it is fast, and because the Linux fonts (`fonts-dejavu`, `fonts-liberation`, installed in setup) differ from the macOS ones anyway. `.skyignore` keeps `data/`, `checkpoints/`, `runs/`, and `outputs/` out of the upload.

Watch it live — SkyPilot registers the cluster as an SSH alias:

```bash
ssh -L 6006:localhost:6006 ocr -t 'cd ~/sky_workdir && tensorboard --logdir runs'
```

then open <http://localhost:6006>. `sky logs ocr` tails the training stdout without the tunnel.

Bring the results back and shut the box down:

```bash
remote/fetch_results.sh        # rsyncs checkpoints/ and runs/ into this repo
sky down ocr
```

Pods bill while they exist. If you would rather not remember `sky down`, launch with `sky launch --down ...` to tear down when the job finishes, or `sky autostop ocr -i 15 --down` to stop after 15 idle minutes.

Change the GPU by editing `accelerators:` in the two launch files (`sky show-gpus --cloud runpod` lists what is available). Tweak epochs, batch size, or the data count in the `run:` / `setup:` blocks the same way — they are ordinary `main.py` invocations.

## Fine-Tune On Degraded Data

Prepare manifests with the same shape as the synthetic manifests:

Recognition manifest:

```json
{
  "samples": [
    {"image": "line_crops/example.png", "text": "Exact visual transcript"}
  ]
}
```

Detection manifest:

```json
{
  "pages": [
    {
      "image": "pages/example.png",
      "boxes": [[10, 20, 400, 48], [10, 58, 390, 86]]
    }
  ]
}
```

Then load the synthetic checkpoint and continue training:

```bash
python main.py --mode train_recognizer --line_manifest target_lines.json --data_root target_data --recognizer_ckpt checkpoints/recognizer_last.pt --output_dir checkpoints_target
python main.py --mode train_detector --detector_manifest target_pages.json --data_root target_data --detector_ckpt checkpoints/detector_last.pt --output_dir checkpoints_target
```

## OCR A Folder

Full-page OCR with detector:

```bash
python main.py \
  --mode ocr_folder \
  --input_dir data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir outputs/ocr_outputs \
  --detector_ckpt checkpoints/detector_last.pt \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --save_visualizations \
  --save_crops
```

Cropped-image OCR without detector:

```bash
python main.py \
  --mode ocr_folder \
  --input_dir data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir outputs/ocr_outputs_crops \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --skip_detector
```

Outputs:

- `texts/<image>.txt`
- `predictions.json`
- optional `crops/<image>/region_*.png`
- optional `visualizations/*_boxes.png` and `*_prob.png`

Training, generation, validation, OCR, and benchmark modes print concise progress with ETA. Training/validation logs include loss plus task metrics: detector precision/recall/F1, recognizer CER/WER/exact match where labels exist. Training modes also log to TensorBoard (see *Watching Training Live*). Folder OCR logs image progress and region counts; OCR accuracy is reported as unavailable unless you provide transcripts.

## Benchmark

If OCR has already been run:

```bash
python main.py \
  --mode benchmark \
  --predictions_json outputs/ocr_outputs/predictions.json \
  --labels_manifest labels.json
```

If no labels are provided, benchmark mode still reports the number of predictions and states that GT metrics were not computed.

To run inference and benchmark in one command:

```bash
python main.py \
  --mode benchmark \
  --input_dir data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir outputs/ocr_outputs \
  --detector_ckpt checkpoints/detector_last.pt \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --labels_manifest labels.json
```

## Smoke Test

```bash
python main.py --mode smoke_test --output_dir outputs/smoke
```

This generates two small synthetic pages and checks detector/recognizer forward passes.

## Notes

- PNG is the primary format throughout generation and inference.
- JPG/JPEG/TIF/TIFF can be read for convenience.
- Detector training can start from synthetic annotations or from a checkpoint.
- Use `--skip_detector` whenever inputs are already cropped text images.
- The provided NoisyOffice folder is an enhancement/binarization dataset. It does not include OCR transcripts, so OCR CER/WER require you to provide text labels separately.

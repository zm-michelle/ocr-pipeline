# Degraded Printed Document OCR

Optical character recognition for scanned printed documents in poor condition
(stains, folds, wrinkles, faded ink, low contrast). Two PyTorch models are
trained on synthetic pages and fine-tuned on real scans. Output is literal: no
spell checking, lexicon, or language model is applied at any stage.

## Architecture

```
page image -> detector -> line boxes -> crops -> recognizer -> text
```

| Stage | Model | Input | Output |
| --- | --- | --- | --- |
| Detection | DBNet-style segmentation, 0.5M parameters | 540x258 grayscale page | text probability and threshold maps |
| Recognition | CRNN + BiLSTM + CTC, 7.6M parameters | 32x1280 grayscale line crop | character sequence over a 95-character set |

The detector is trained with the differentiable binarization objective (Liao et
al., AAAI 2020): targets are shrunk boxes, a second head predicts a per-pixel
threshold, and the loss combines balanced cross-entropy, Dice, and an L1 term
on the threshold map. Detected boxes are expanded back to line size using the
predicted threshold map. `--detector_loss bce` selects the simpler per-pixel
objective instead.

Line boxes are sorted into reading order; tall regions are split by horizontal
ink projection as a fallback.

## Process

1. **Data generation.** Synthetic pages are rendered from a 20,000-sentence
   public-domain corpus with numbers, dates, codes, and symbols inserted, using
   a randomly selected typeface per page, then degraded (stains, rings, folds,
   noise, fading). Manifests pair each line crop with its exact transcript.
2. **Training.** Detector and recognizer are trained independently on the
   generated data with a cosine learning rate schedule.
3. **Pseudo-labelling.** Real scans without transcripts are labelled by running
   Tesseract on their pixel-aligned clean counterparts and transferring the
   result to the degraded versions.
4. **Fine-tuning.** Both models are fine-tuned on the real labelled pages mixed
   with a sample of synthetic data.
5. **Evaluation.** The full pipeline is scored end to end on held-out pages
   using corpus character and word error rate, and compared against Tesseract.

## Requirements

- Python 3.12 (later versions lack PyTorch wheels)
- Dependencies in `requirements.txt`: PyTorch, torchvision, OpenCV (headless),
  Pillow, NumPy, FastAPI, Uvicorn, TensorBoard
- Optional: `tesseract` for baseline comparison and pseudo-labelling;
  `skypilot[runpod]` for cloud training

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below assume the virtual environment is active. Execution defaults
to CPU; pass `--device cuda` or `--device mps` for GPU.

## Usage

### Web application

```bash
python -m web.server
```

Serves an interface at <http://127.0.0.1:8000> for reading uploaded, pasted, or
browsed images. Displays detected line boxes over the image alongside the
recognized text, and reports character and word error rate where transcripts
are available. The loaded checkpoint paths are shown in the page header.

Options: `--recognizer_ckpt`, `--detector_ckpt`, `--samples_dir`, `--no_samples`,
`--no_detector`, `--device`, `--host`, `--port`.

### Command line

All operations use `python main.py --mode <mode>`.

| Mode | Purpose |
| --- | --- |
| `generate_synth_data` | Render synthetic training pages and manifests |
| `train_detector` | Train the line detector |
| `train_recognizer` | Train the line recognizer |
| `test_detector` | Score the detector against a manifest |
| `test_recognizer` | Score the recognizer against a manifest |
| `eval_e2e` | Score the full pipeline, with an error breakdown |
| `compare` | Score the pipeline against Tesseract and other checkpoints |
| `benchmark` | Score an existing `predictions.json` against labels |
| `ocr_folder` | Transcribe a directory of images |
| `pseudo_label` | Label real scans from clean counterparts |
| `smoke_test` | Verify the installation |

**Transcribe a directory.**

```bash
python main.py --mode ocr_folder \
  --input_dir  data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir outputs/ocr \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --detector_ckpt   checkpoints/detector_last.pt
```

Writes one text file per image, a `predictions.json` containing line boxes, and
optional crops and overlays with `--save_crops` and `--save_visualizations`.
Use `--skip_detector` when inputs are already single line crops.

**Generate training data.**

```bash
python main.py --mode generate_synth_data \
  --output_dir data/synthetic_docs \
  --num_samples 40000 --val_split 0.05 --workers 8
```

`--val_split` holds out a fraction of pages for validation. Pages are seeded
individually, so output is identical for any `--workers` value. On Linux, run
`remote/fonts.sh` first to install typefaces without root privileges.

**Train.**

```bash
python main.py --mode train_recognizer \
  --line_manifest     data/synthetic_docs/lines_manifest_train.json \
  --val_line_manifest data/synthetic_docs/lines_manifest_val.json \
  --data_root data/synthetic_docs --output_dir checkpoints \
  --epochs 14 --batch_size 64 --augment

python main.py --mode train_detector \
  --detector_manifest     data/synthetic_docs/pages_manifest_train.json \
  --val_detector_manifest data/synthetic_docs/pages_manifest_val.json \
  --data_root data/synthetic_docs --output_dir checkpoints \
  --epochs 16 --batch_size 16 --augment
```

Add `--device cuda --amp --pin_memory --num_workers 6` on a GPU. Metrics are
written to `runs/<name>/` for TensorBoard (`tensorboard --logdir runs`) and to
`metrics.jsonl`.

**Label real scans.**

```bash
python main.py --mode pseudo_label \
  --clean_dir  data/SimulatedNoisyOffice/clean_images_grayscale_doubleresolution \
  --input_dir  data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir data/noisyoffice_labels
```

**Fine-tune.** Manifest arguments accept comma-separated mixes, where `:8`
repeats a file and `:0.1` samples a random tenth.

```bash
python main.py --mode train_recognizer \
  --line_manifest     "data/synthetic_docs/lines_manifest_train.json:0.1,data/noisyoffice_labels/lines_manifest_train.json:8" \
  --val_line_manifest data/noisyoffice_labels/lines_manifest_test.json \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --output_dir checkpoints/finetuned \
  --epochs 6 --lr 5e-5 --augment
```

**Evaluate.**

```bash
python main.py --mode compare \
  --detector_manifest data/noisyoffice_labels/pages_manifest_test.json \
  --data_root data/noisyoffice_labels \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --detector_ckpt   checkpoints/detector_last.pt \
  --output_dir outputs/comparison
```

Scores the current checkpoints against Tesseract on identical pages. Additional
models are added with `--compare_model "name=recognizer.pt:detector.pt"`,
repeatable. `--mode eval_e2e` scores a single system and additionally reports
error rate by font, style, and line length, common character confusions, and
detector misses against spurious boxes.

### Remote training

A Slurm driver is provided. Copy `remote/hpc.env.example` to `remote/hpc.env`
and set the host, user, partition, and account; SSH key access is required.

```bash
remote/hpc.sh push | setup | submit | status | logs | fetch
```

The job generates data, trains both models, evaluates, fine-tunes on real pages
when `REAL_DIR` is set, and evaluates again. Stages are selectable
(`remote/hpc.sh submit detector,e2e`). `DATA_DIR` should point at a project
filesystem rather than a home directory, as a 40,000-page dataset contains
approximately 380,000 files.

SkyPilot configurations for RunPod are in `remote/recognizer.sky.yaml` and
`remote/detector.sky.yaml`.

## Results

Corpus error rates on 72 held-out real scans from SimulatedNoisyOffice,
reproducible with `--mode compare`:

| System | CER | WER | s/page |
| --- | --- | --- | --- |
| Current model, fine-tuned | 4.00% | 10.74% | 0.29 |
| Current model, synthetic training only | 5.29% | 16.36% | 0.28 |
| Tesseract 5.5.3 | 7.66% | 17.60% | 0.35 |
| First cluster run | 30.64% | 63.80% | 0.20 |
| Initial model | 54.56% | 81.59% | 0.15 |

Reference transcripts are produced by Tesseract on the clean counterpart of
each page and are therefore an approximation rather than a human transcription.

## Repository layout

```
main.py                command line entry point
ocr/
  config.py            character set, image sizes, typeface lists
  ctc.py               character encoding and CTC decoding
  cli.py               argument parsing and mode dispatch
  models/              detector and recognizer definitions
  data/                datasets, transforms, synthetic generation,
                       detector targets, dataset catalogue, pseudo-labelling
  inference/           detection postprocessing, recognition, OCRPipeline
  training/            training loops, losses, checkpoints, metric logging
  evaluation/          metrics, end-to-end scoring, error breakdown, comparison
web/                   browser application
remote/                cluster and cloud training
tests/                 python tests/test_synthetic_boxes.py
checkpoints/           model weights (untracked)
data/                  datasets (untracked, except the text corpus)
runs/                  training logs (untracked)
outputs/               transcripts and reports (untracked)
```

Supported image formats: PNG (primary), JPG, JPEG, TIF, TIFF.

# Degraded Printed Document OCR

Reads text out of scanned printed pages that are in poor condition: coffee
stains, folds, wrinkles, faded ink, low contrast.

Two small PyTorch models do the work.

```
full page  ->  DBNet-style detector finds the text lines
           ->  each line cropped out
           ->  CRNN + CTC recognizer reads each crop
           ->  plain text
```

Output is literal. There is no spell checker, dictionary, or language model
anywhere in the pipeline, so a half-visible word comes out half-visible rather
than guessed at.

## Where it stands

On 72 held-out real scans from the SimulatedNoisyOffice corpus:

```
system                         CER       WER    s/page
v3 fine-tuned (current)      4.00%    10.74%     0.29s
v3 synthetic-only            5.29%    16.36%     0.28s
tesseract 5.5.3              7.66%    17.60%     0.35s
v1 first cluster run        30.64%    63.80%     0.20s
original (pre-session)      54.56%    81.59%     0.15s
```

CER is character error rate, lower is better. Reproduce this table yourself
with `--mode compare`, described under "Measuring how good it is" below.

One caveat worth carrying: those labels come from running Tesseract on the
clean version of each page, not from a human. So 4% means "4% different from
what Tesseract read on the clean scan", which is a good proxy but not truth.

## Setting up

Needs Python 3.12. Newer versions do not have PyTorch wheels yet.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every command below assumes that venv is active. Runs on CPU; add
`--device cuda` (or `--device mps` on a Mac) when you have a GPU.

Some optional extras:

- `brew install tesseract` (or `apt install tesseract-ocr`) for the baseline
  comparison and for labelling real scans.
- `pip install "skypilot[runpod]"` only if you want to train on rented GPUs.

## Reading some pages

The quickest way to see what it does is the web app:

```bash
python -m web.server
```

Open <http://127.0.0.1:8000>. Drag an image onto the page, click to browse for
one, or paste one from the clipboard. You get the detected line boxes drawn
over your image, the text beside it, and hovering either side highlights the
matching line.

The header tells you exactly which checkpoint files are loaded, so you always
know which model you are looking at. To try a different one, restart with
`--detector_ckpt path/to/detector.pt` and `--recognizer_ckpt path/to/rec.pt`.

Three controls re-run the current image without reloading anything:

- **Detect lines** - turn it off when your image is already a single cropped line.
- **Learned threshold** - lets the detector use the cutoff it learned instead of
  a fixed one. On by default for models trained with the DB objective. Leave it on.
- **Threshold** - the fixed cutoff, used only when the learned one is off.

Below the drop zone is a browser over every image folder under `data/`. Pick a
collection, filter by filename, then click a thumbnail to read it or drag one
onto the drop zone. Nothing uploads; the server opens the file directly. Where
a dataset ships transcripts, the expected text and a live CER come back with
the prediction, and those collections are marked in the dropdown.

When a dataset has several versions of one page, they are linked. NoisyOffice
pairs every degraded scan with clean originals, so a "Same page" strip appears
under the image letting you flip between them. That is the fastest way to see
how much of an error is the damage and how much is the model.

### From the command line

For a whole folder at once:

```bash
python main.py --mode ocr_folder \
  --input_dir  data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir outputs/ocr \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --detector_ckpt   checkpoints/detector_last.pt \
  --save_visualizations
```

You get one `.txt` per image under `texts/`, a `predictions.json` with the line
boxes, and with `--save_visualizations` an overlay image showing what was
detected. Add `--skip_detector` if the inputs are already single cropped lines.

### From Python

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

## Measuring how good it is

Three commands, increasing in usefulness.

**Against other systems.** This is the one to reach for. It runs everything
over the same pages and prints a table:

```bash
python main.py --mode compare \
  --detector_manifest data/noisyoffice_labels/pages_manifest_test.json \
  --data_root data/noisyoffice_labels \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --detector_ckpt   checkpoints/detector_last.pt \
  --output_dir outputs/comparison
```

By default that scores your current checkpoints against Tesseract. Add older
models to the same table with `--compare_model`, repeated as often as you like:

```bash
  --compare_model "v3 synthetic-only=checkpoints/v3/recognizer_last.pt:checkpoints/v3/detector_last.pt" \
  --compare_model "v1=checkpoints/v1/recognizer_v1.pt:checkpoints/v1/detector_v1.pt"
```

Other flags: `--label` names your current model in the table, `--baselines ""`
drops Tesseract, `--no_self` leaves your own model out, `--limit N` scores only
the first N pages while you are iterating. Results land in
`comparison.json` and `comparison.txt`.

**End to end, with a breakdown.** Same scoring, one system, plus detail on
where the errors are:

```bash
python main.py --mode eval_e2e \
  --detector_manifest data/noisyoffice_labels/pages_manifest_test.json \
  --data_root data/noisyoffice_labels \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --detector_ckpt   checkpoints/detector_last.pt \
  --output_dir outputs/e2e
```

Alongside the totals you get CER split by font, style and line length, the most
common character confusions, how many lines the detector missed versus how many
spurious boxes it invented, and the worst pages by name. That breakdown is what
tells you whether to work on the detector or the recognizer next.

**One component at a time.** `--mode test_recognizer` and `--mode test_detector`
score a single model against a manifest. Useful for isolating a regression, but
remember that both can improve while the pipeline gets worse, which is exactly
why `compare` and `eval_e2e` exist.

## Making training data

There are no real labelled pages to train on, so the pipeline generates its own.

```bash
python main.py --mode generate_synth_data \
  --output_dir data/synthetic_docs \
  --num_samples 40000 --val_split 0.05 --workers 8
```

This writes page images, line crops, and manifests pairing each crop with its
exact text. `--val_split 0.05` holds out 5% for validation, split by page so no
line from a validation page can leak into training. `--workers` renders in
parallel; pages are seeded individually, so the output is identical no matter
how many workers you use.

Text comes from `ocr/data/corpus/english.txt`, about 20,000 sentences from
public-domain novels, with numbers, dates, prices, reference codes, names and
symbols mixed in so the recognizer sees every character it might meet.

Each page picks a random typeface from whatever is installed. On a Mac it finds
around 25; on a bare Linux box run `remote/fonts.sh` first to install DejaVu,
Liberation and FreeFont without needing root. The generator prints what it
found and refuses to run if it would fall back to a bitmap font.

Pages get coffee stains, cup rings, folds, wrinkles, noise and fading. Some are
two-column, some are slightly skewed. The aim is to look like the real scans,
and getting that right mattered far more than anything else in this project.

## Training

Recognizer:

```bash
python main.py --mode train_recognizer \
  --line_manifest     data/synthetic_docs/lines_manifest_train.json \
  --val_line_manifest data/synthetic_docs/lines_manifest_val.json \
  --data_root data/synthetic_docs --output_dir checkpoints \
  --epochs 14 --batch_size 64 --augment
```

Detector:

```bash
python main.py --mode train_detector \
  --detector_manifest     data/synthetic_docs/pages_manifest_train.json \
  --val_detector_manifest data/synthetic_docs/pages_manifest_val.json \
  --data_root data/synthetic_docs --output_dir checkpoints \
  --epochs 16 --batch_size 16 --augment
```

On a GPU add `--device cuda --amp --pin_memory --num_workers 6`. Both use a
cosine learning rate schedule by default, warming up then decaying.

Always pass a validation manifest. Without one you get training curves only,
and none of the sample predictions or mask images that make a run readable.

### A note on the detector objective

By default the detector trains the DBNet way rather than on plain per-pixel
loss. The problem with per-pixel loss is that a pixel in the small gap between
two lines costs almost nothing to get wrong, so the model learns a blurry mask,
the gaps fill in, and two lines merge into one box. The recognizer then gets a
crop with two lines stacked in it and returns nonsense.

The fix has three parts: shrink the target boxes so the gap between lines is
wide and unambiguous, have the model predict its own per-pixel cutoff instead
of using a fixed one, and score overlap of regions rather than counting pixels.
`--detector_loss bce` goes back to the old behaviour if you want to compare.

Validation reports three F1 scores. Pixel F1 will happily give a high score to
a model that paints eight lines as one blob. Box F1 at IoU 0.5 will not, and
`detector_only` box F1 shows what the network manages without a downstream
line-splitting heuristic rescuing it. Watch the last two.

## Watching training

Every training run writes to `runs/<name>/`.

```bash
tensorboard --logdir runs
```

Open <http://localhost:6006>. You get loss and metric curves, and for the
recognizer a table of sample predictions next to their targets, refreshed each
epoch. That table is worth watching: CTC outputs nothing but blanks for the
first few hundred steps, then fragments, then words, and the CER number looks
identical through the first two of those.

Each run also writes `metrics.jsonl` and `hparams.json` in plain text, so the
numbers are readable without TensorBoard.

## Using real scans

NoisyOffice ships no transcripts, but every damaged page has a clean twin. This
reads the clean ones with Tesseract and applies the result to the damaged ones:

```bash
python main.py --mode pseudo_label \
  --clean_dir  data/SimulatedNoisyOffice/clean_images_grayscale_doubleresolution \
  --input_dir  data/SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir data/noisyoffice_labels
```

That gives 216 labelled real pages, split by the dataset's own naming into 144
for training and 72 for testing. The labels carry some noise where Tesseract
misread something, so treat them as a good proxy rather than ground truth.

Fine-tuning on them is what took the current model from 5.29% to 4.00%.
Manifest arguments accept mixes, where `:8` repeats a file eight times and
`:0.1` takes a random tenth of it:

```bash
python main.py --mode train_recognizer \
  --line_manifest     "data/synthetic_docs/lines_manifest_train.json:0.1,data/noisyoffice_labels/lines_manifest_train.json:8" \
  --val_line_manifest data/noisyoffice_labels/lines_manifest_test.json \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --output_dir checkpoints/finetuned \
  --epochs 6 --lr 5e-5 --augment
```

Repeating the small real set against a slice of the large synthetic one keeps
the model from forgetting what it already knew.

## Training somewhere else

Full runs take a few hours, which is a lot on a laptop. Two paths are set up.

**A Slurm cluster.** Copy `remote/hpc.env.example` to `remote/hpc.env` and fill
in your host, username, partition and account. You need SSH key access to the
login node first (`ssh-copy-id yourhost`). Then:

```bash
remote/hpc.sh push      # copy the code over
remote/hpc.sh setup     # build a venv and install fonts, once
remote/hpc.sh submit    # queue the job
remote/hpc.sh status    # is it running yet
remote/hpc.sh logs      # follow the output
remote/hpc.sh fetch     # bring checkpoints and runs back
```

The job generates data, trains both models, evaluates them, fine-tunes on real
pages if `REAL_DIR` is set, and evaluates again. Pick stages with
`remote/hpc.sh submit detector,e2e`. Override sizes at submit time with
`NUM_SAMPLES`, `DATA_DIR`, `EPOCHS_REC`, `EPOCHS_DET`, `EPOCHS_FT`.

Put `DATA_DIR` on the cluster's large project filesystem, not your home
directory. A 40,000-page dataset is about 380,000 files and home quotas are
usually too small for it.

For TensorBoard on the cluster, tunnel it:

```bash
ssh -L 6006:localhost:6006 yourhost -t 'cd ~/ocr_project && .venv/bin/tensorboard --logdir runs'
```

**Rented GPUs via SkyPilot.** `remote/recognizer.sky.yaml` and
`remote/detector.sky.yaml` target a RunPod RTX 4090. Store your key with
`runpod config`, then `sky launch -c ocr remote/recognizer.sky.yaml`. Remember
`sky down ocr` afterwards, since pods bill while they exist.

## What is where

```
main.py                  command line entry point
ocr/
  config.py              charset, image sizes, font lists
  ctc.py                 text encoding and CTC decoding
  cli.py                 every --mode lives here
  models/                the detector and recognizer networks
  data/                  datasets, transforms, synthetic generation,
                         DBNet targets, dataset browsing, pseudo-labelling
  inference/             detection postprocessing, recognition, OCRPipeline
  training/              training loops, losses, checkpoints, TensorBoard
  evaluation/            metrics, end-to-end scoring, error breakdown,
                         comparison against other systems
web/                     the browser app
remote/                  cluster and cloud training
tests/                   run with: python tests/test_synthetic_boxes.py
checkpoints/             trained weights (not in git)
data/                    datasets (not in git except the corpus)
runs/                    TensorBoard logs (not in git)
outputs/                 OCR results and reports (not in git)
```

## Things that will trip you up

- **Always look at an end-to-end number before believing a component number.**
  The detector's pixel score and the recognizer's line score can both improve
  while the actual output gets worse. This happened twice here.
- **Synthetic accuracy is not real accuracy.** One change improved synthetic CER
  by 43% and made real scans 29% worse. Test on real pages.
- **Generation scales differently than you expect.** A bug that appears once in
  6,000 pages will not show up in a 50-page trial run. `tests/` exists because
  of one such bug that killed a 40,000-page run partway through.
- On a cluster, cap BLAS threads (`OMP_NUM_THREADS`) or imports segfault on
  high-core-count login nodes. `remote/hpc.sh` does this for you.
- PNG is the main format. JPG, TIF and TIFF can be read too.

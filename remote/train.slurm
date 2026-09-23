#!/usr/bin/env bash
# Slurm job: generate synthetic data (once) and train both models on one GPU.
#
# Submit with remote/hpc.sh (it passes partition/account/time from hpc.env as
# sbatch flags, which override the defaults below), or by hand:
#   sbatch --partition=<gpu-partition> --gres=gpu:1 remote/train.slurm
#
# Everything below runs on the compute node. It assumes `remote/hpc.sh setup`
# already built .venv and installed fonts on the login node, because compute
# nodes often have no internet.

#SBATCH --job-name=ocr-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1   # progress lines reach the .out file as they happen, not in 8 KB chunks
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p runs

# HPC_MODULES comes from remote/hpc.env via `hpc.sh submit` (sbatch --export).
# Loaded here too because the compute node needs the same interpreter the venv
# was built against on the login node.
if [ -n "${HPC_MODULES:-}" ]; then
  # Batch jobs do not always start as login shells; make sure Lmod is initialised.
  if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
  fi
  echo "== module load $HPC_MODULES"
  module load $HPC_MODULES
fi

# The python/gpu module also puts ITS cuDNN/CUDA/NCCL on LD_LIBRARY_PATH (cuDNN
# 9.3 on Quartz). Pip torch bundles its own newer cuDNN and must load that one,
# or it dies with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. Drop those entries but
# keep the module's lib/ dir, which the interpreter needs for libpython.
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '/cuda(/|$)|/nccl(/|$)|/nvidia/' | paste -sd: -)"

source .venv/bin/activate
export OCR_FONT_DIRS="$HOME/.local/share/fonts/ocr"
# Size BLAS/OpenMP pools to the CPUs Slurm gave us, not the node's 128 cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no nvidia-smi)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DEVICE="${DEVICE:-cuda}"          # DEVICE=cpu lets you dry-run this script on a laptop
NUM_SAMPLES="${NUM_SAMPLES:-20000}"
DATA_DIR="${DATA_DIR:-data/synthetic_docs}"   # a new dir per generator/font change keeps old runs comparable
EPOCHS_REC="${EPOCHS_REC:-30}"
EPOCHS_DET="${EPOCHS_DET:-20}"
DETECTOR_LOSS="${DETECTOR_LOSS:-db}"   # db (DBNet objective) or bce (original)
REAL_DIR="${REAL_DIR:-}"                       # pseudo-labelled real pages (ocr/data/pseudo_label.py); enables finetune + e2e_real
EPOCHS_FT="${EPOCHS_FT:-6}"
STAGES="${STAGES:-recognizer,detector,e2e,finetune,e2e_real}"   # list; e.g. STAGES=detector,e2e to reuse a trained recognizer
STAGES="${STAGES//+/,}"                       # hpc.sh sends a+b because sbatch --export splits on commas
WORKERS="${SLURM_CPUS_PER_TASK:-6}"
JOB="${SLURM_JOB_ID:-local}"

stage() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

if [ ! -f $DATA_DIR/lines_manifest_train.json ]; then
  # Cheap guard: the generator once died 11k pages into a 40k run on one bad box.
  echo "== generator self-test"
  python tests/test_synthetic_boxes.py
  echo "== generating $NUM_SAMPLES synthetic pages"
  python main.py --mode generate_synth_data \
    --output_dir "$DATA_DIR" --num_samples "$NUM_SAMPLES" --val_split 0.05 --preview_count 12 --workers "$WORKERS"
fi

if stage recognizer; then
  echo "== training recognizer"
  python main.py --mode train_recognizer \
    --line_manifest     $DATA_DIR/lines_manifest_train.json \
    --val_line_manifest $DATA_DIR/lines_manifest_val.json \
    --data_root "$DATA_DIR" --output_dir checkpoints --run_name "recognizer-$JOB" \
    --device "$DEVICE" --amp --pin_memory --num_workers "$WORKERS" \
    --batch_size 64 --epochs "$EPOCHS_REC" --lr 3e-4 --augment --log_every 50
fi

if stage detector; then
  echo "== training detector ($DETECTOR_LOSS loss)"
  python main.py --mode train_detector --detector_loss "$DETECTOR_LOSS" \
    --detector_manifest     $DATA_DIR/pages_manifest_train.json \
    --val_detector_manifest $DATA_DIR/pages_manifest_val.json \
    --data_root "$DATA_DIR" --output_dir checkpoints --run_name "detector-$DETECTOR_LOSS-$JOB" \
    --device "$DEVICE" --amp --pin_memory --num_workers "$WORKERS" \
    --batch_size 16 --epochs "$EPOCHS_DET" --lr 3e-4 --augment --log_every 50
fi

if stage e2e; then
  # The number that describes the product: page in, text out, on held-out pages.
  echo "== end-to-end evaluation on validation pages"
  python main.py --mode eval_e2e \
    --detector_manifest $DATA_DIR/pages_manifest_val.json --data_root "$DATA_DIR" \
    --detector_ckpt checkpoints/detector_last.pt --recognizer_ckpt checkpoints/recognizer_last.pt \
    --output_dir "runs/e2e-$JOB" --run_name "e2e-$JOB" --device "$DEVICE" --log_every 100
fi

if stage finetune && [ -n "$REAL_DIR" ] && [ -f "$REAL_DIR/lines_manifest_train.json" ]; then
  # Real pages are few (144), so mix them, repeated, with a slice of synthetic: the
  # model keeps what it knows and learns what real degradation looks like.
  echo "== fine-tuning on real + synthetic (real x8, synthetic 10%)"
  mkdir -p checkpoints/finetuned
  python main.py --mode train_recognizer \
    --line_manifest     "$DATA_DIR/lines_manifest_train.json:0.1,$REAL_DIR/lines_manifest_train.json:8" \
    --val_line_manifest "$REAL_DIR/lines_manifest_test.json" \
    --recognizer_ckpt checkpoints/recognizer_last.pt --output_dir checkpoints/finetuned --run_name "finetune-recognizer-$JOB" \
    --device "$DEVICE" --amp --pin_memory --num_workers "$WORKERS" \
    --batch_size 64 --epochs "$EPOCHS_FT" --lr 5e-5 --warmup_steps 50 --augment --log_every 50
  python main.py --mode train_detector --detector_loss "$DETECTOR_LOSS" \
    --detector_manifest     "$DATA_DIR/pages_manifest_train.json:0.1,$REAL_DIR/pages_manifest_train.json:6" \
    --val_detector_manifest "$REAL_DIR/pages_manifest_test.json" \
    --detector_ckpt checkpoints/detector_last.pt --output_dir checkpoints/finetuned --run_name "finetune-detector-$JOB" \
    --device "$DEVICE" --amp --pin_memory --num_workers "$WORKERS" \
    --batch_size 16 --epochs "$EPOCHS_FT" --lr 1e-4 --warmup_steps 50 --augment --log_every 50
fi

if stage e2e_real && [ -n "$REAL_DIR" ] && [ -f "$REAL_DIR/pages_manifest_test.json" ]; then
  echo "== end-to-end on REAL test pages: before and after fine-tuning"
  python main.py --mode eval_e2e --detector_manifest "$REAL_DIR/pages_manifest_test.json" --data_root "$REAL_DIR" \
    --detector_ckpt checkpoints/detector_last.pt --recognizer_ckpt checkpoints/recognizer_last.pt \
    --output_dir "runs/e2e-real-$JOB" --run_name "e2e-real-$JOB" --device "$DEVICE" --log_every 1000 | sed 's/^E2E/E2E-REAL (synthetic-only model)/'
  if [ -f checkpoints/finetuned/recognizer_last.pt ]; then
    python main.py --mode eval_e2e --detector_manifest "$REAL_DIR/pages_manifest_test.json" --data_root "$REAL_DIR" \
      --detector_ckpt checkpoints/finetuned/detector_last.pt --recognizer_ckpt checkpoints/finetuned/recognizer_last.pt \
      --output_dir "runs/e2e-real-ft-$JOB" --run_name "e2e-real-ft-$JOB" --device "$DEVICE" --log_every 1000 | sed 's/^E2E/E2E-REAL (fine-tuned model)/'
  fi
fi

echo "== done $(date)"

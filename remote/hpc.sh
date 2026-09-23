#!/usr/bin/env bash
# Drive training on a Slurm cluster from your laptop.
#
#   remote/hpc.sh push         rsync the code up (honours .skyignore)
#   remote/hpc.sh setup        build .venv + install fonts on the login node (once)
#   remote/hpc.sh submit       sbatch remote/train.slurm
#   remote/hpc.sh status       your jobs in the queue
#   remote/hpc.sh logs [JOB]   follow a job's output (latest if omitted)
#   remote/hpc.sh tensorboard  tunnel port 6006 and run TensorBoard on the login node
#   remote/hpc.sh fetch        rsync checkpoints/ and runs/ back down
#   remote/hpc.sh cancel JOB
#
# Configure via remote/hpc.env (see hpc.env.example) or the same names as env vars.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in ""|-h|--help) sed -n '2,13p' "$0"; exit 0;; esac

# hpc.env supplies DEFAULTS: a variable already set in the environment
# (DATA_DIR=... remote/hpc.sh submit) must win over the file, not be overwritten by it.
if [ -f remote/hpc.env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"                       # strip comments
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; key="${key//[[:space:]]/}"
    val="${line#*=}"; val="${val#"${val%%[![:space:]]*}"}"; val="${val%"${val##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    if [ -z "${!key:-}" ]; then export "$key=$val"; fi
  done < remote/hpc.env
fi
: "${HPC_HOST:?set HPC_HOST (see remote/hpc.env.example)}"
: "${HPC_USER:?set HPC_USER}"
HPC_DIR="${HPC_DIR:-ocr_project}"
# Relative -> under the cluster home dir; absolute (starts with /) -> used as-is,
# for big scratch/slate filesystems where a 200k-image dataset belongs.
case "$HPC_DIR" in /*) REMOTE_DIR="$HPC_DIR";; *) REMOTE_DIR="~/$HPC_DIR";; esac
HPC_PARTITION="${HPC_PARTITION:-gpu}"
HPC_TIME="${HPC_TIME:-08:00:00}"
HPC_GRES="${HPC_GRES:-gpu:1}"
HPC_PYTHON="${HPC_PYTHON:-python3}"
TARGET="$HPC_USER@$HPC_HOST"

# Run a command on the login node inside a LOGIN shell (bash -l): that is what
# defines the `module` function on Lmod clusters; a plain `ssh host cmd` does not.
# The script is fed on stdin so no quoting gymnastics are needed.
# BLAS_CAP: a 128-core login node makes OpenBLAS spawn a thread per core on
# import, which trips the per-user process limit and segfaults numpy/cv2/torch.
BLAS_CAP="export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"
remote() {
  ssh -o BatchMode=yes "$TARGET" bash -l <<EOF
cd $REMOTE_DIR 2>/dev/null || true
$BLAS_CAP
$*
EOF
}
with_modules() { # prefix a command with the module loads, if any
  if [ -n "${HPC_MODULES:-}" ]; then echo "module load $HPC_MODULES && $*"; else echo "$*"; fi
}

case "${1:-}" in
  push)
    echo "-> rsync . -> $TARGET:$REMOTE_DIR (excludes from .skyignore)"
    rsync -az --progress --exclude-from=.skyignore --exclude=remote/hpc.env ./ "$TARGET:$REMOTE_DIR/"
    ;;
  setup)
    echo "-> building .venv and fonts on the login node"
    remote "$(with_modules "$HPC_PYTHON -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt && PYTHON=.venv/bin/python remote/fonts.sh")"
    ;;
  submit)
    ACCT=""; [ -n "${HPC_ACCOUNT:-}" ] && ACCT="--account=$HPC_ACCOUNT"
    echo "-> sbatch on $HPC_PARTITION ($HPC_GRES, $HPC_TIME)"
    # DATA_DIR/NUM_SAMPLES/EPOCHS_* come from the environment or hpc.env; DATA_DIR should be on big storage.
    # sbatch --export splits on commas, so stages travel as a+b+c and train.slurm turns them back
    STAGES_ARG="${2:-recognizer,detector,e2e,finetune,e2e_real}"
    echo "   stages: $STAGES_ARG   detector loss: ${DETECTOR_LOSS:-db}   data: ${DATA_DIR:-data/synthetic_docs} (${NUM_SAMPLES:-20000} pages)   epochs rec/det/ft: ${EPOCHS_REC:-30}/${EPOCHS_DET:-20}/${EPOCHS_FT:-6}   real: ${REAL_DIR:-none}"
    remote "mkdir -p runs && sbatch --partition=$HPC_PARTITION --gres=$HPC_GRES --time=$HPC_TIME $ACCT --export=ALL,HPC_MODULES='${HPC_MODULES:-}',STAGES='${STAGES_ARG//,/+}',DETECTOR_LOSS='${DETECTOR_LOSS:-db}',NUM_SAMPLES='${NUM_SAMPLES:-20000}',DATA_DIR='${DATA_DIR:-data/synthetic_docs}',EPOCHS_REC='${EPOCHS_REC:-30}',EPOCHS_DET='${EPOCHS_DET:-20}',EPOCHS_FT='${EPOCHS_FT:-6}',REAL_DIR='${REAL_DIR:-}' remote/train.slurm"""""
    ;;
  fonts)
    echo "-> refreshing fonts on the login node"
    remote "$(with_modules "PYTHON=.venv/bin/python bash remote/fonts.sh")"
    ;;
  status)
    remote "squeue -u $HPC_USER -o '%.10i %.12j %.9P %.8T %.10M %.6D %R'"
    ;;
  logs)
    JOB="${2:-}"
    if [ -z "$JOB" ]; then
      remote "f=\$(ls -t runs/slurm-*.out 2>/dev/null | head -1); [ -n \"\$f\" ] && tail -n 40 -f \"\$f\" || echo 'no job output yet'"
    else
      remote "tail -n 40 -f runs/slurm-$JOB.out"
    fi
    ;;
  tensorboard)
    echo "-> TensorBoard on the login node, tunnelled to http://localhost:6006  (Ctrl-C to stop)"
    ssh -L 6006:localhost:6006 -t "$TARGET" "bash -lc '$BLAS_CAP; cd $REMOTE_DIR && $(with_modules ".venv/bin/tensorboard --logdir runs --port 6006")'"
    ;;
  fetch)
    mkdir -p checkpoints runs
    rsync -az --progress "$TARGET:$REMOTE_DIR/checkpoints/" ./checkpoints/
    rsync -az --progress "$TARGET:$REMOTE_DIR/runs/"        ./runs/
    echo "-> done; restart the web app for the new checkpoints, tensorboard --logdir runs for the curves"
    ;;
  cancel)
    remote "scancel ${2:?job id}"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1
    ;;
esac

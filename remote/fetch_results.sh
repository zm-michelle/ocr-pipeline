#!/usr/bin/env bash
# Pull trained checkpoints and TensorBoard runs down from the SkyPilot cluster.
#   remote/fetch_results.sh [cluster-name]   (default: ocr)
set -euo pipefail
CLUSTER="${1:-ocr}"
cd "$(dirname "$0")/.."
rsync -avz --progress "$CLUSTER:~/sky_workdir/checkpoints/" ./checkpoints/
rsync -avz --progress "$CLUSTER:~/sky_workdir/runs/"        ./runs/
echo "done — restart the web app to pick up the new checkpoints; tensorboard --logdir runs to see the curves"

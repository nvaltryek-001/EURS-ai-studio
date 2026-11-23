#!/usr/bin/env bash
# Full experiment: split -> train -> evaluate on held-out test -> paper tables/figures + web results.
#   bash ml/run_pipeline.sh data/eurs_train eurs            # official data
#   bash ml/run_pipeline.sh data/synthetic synthetic 300    # synthetic smoke test (generates data)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:?dataset root}; NAME=${2:-eurs}; SYN=${3:-0}
SPLIT=data/splits/$NAME; RES=results/$NAME; TAG=$([ "$NAME" = synthetic ] && echo synthetic || echo main)
EXTRA=$([ "$SYN" != 0 ] && echo "--synthetic $SYN" || echo "")

python ml/prepare_dataset.py --root "$DATA" --out "$SPLIT" $EXTRA
python ml/train_lite.py --manifest "$SPLIT/manifest.csv" --out "runs/${NAME}_lite"
python ml/evaluate.py --manifest "$SPLIT/manifest.csv" --model classical --out "$RES/classical"
python ml/evaluate.py --manifest "$SPLIT/manifest.csv" --model "runs/${NAME}_lite/eurs_lite.npz" --out "$RES/eurs_lite"
LOGS=(--train-log "runs/${NAME}_lite/train_log.json")

if python -c "import torch" 2>/dev/null; then
  python ml/train_net.py --manifest "$SPLIT/manifest.csv" --out "runs/${NAME}_net" ${NET_ARGS:-}
  python ml/export_onnx.py --ckpt "runs/${NAME}_net/best.pt" --out "runs/${NAME}_net/eurs_net.onnx"
  python ml/evaluate.py --manifest "$SPLIT/manifest.csv" --model "runs/${NAME}_net/eurs_net.onnx" --out "$RES/eurs_net"
  LOGS+=(--train-log "runs/${NAME}_net/train_log.json")
else
  echo "PyTorch not installed: skipping EURS-Net (pip install -r ml/requirements-torch.txt)"
fi

python ml/make_report.py --tag "$TAG" --results "$RES" --split-report "$SPLIT/split_report.json" "${LOGS[@]}"
echo "Done. Results in $RES, paper assets in paper/generated + paper/figures."

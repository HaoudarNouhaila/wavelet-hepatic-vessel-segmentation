#!/usr/bin/env bash
# Enchaîne toutes les étapes du projet (Linux, WSL ou Git Bash), dans l'ordre utilisé pour les
# résultats du README. Sur CPU, l'ensemble prend plus d'une journée de calcul.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
THREADS=${THREADS:-2}

$PY scripts/check_environment.py
$PY -m pytest

# Étape A : données
$PY scripts/download_data.py --config configs/data.yaml
$PY scripts/prepare_data.py --config configs/data.yaml
$PY scripts/inspect_dataset.py --config configs/data.yaml
$PY scripts/wavelet_analysis.py --config configs/data.yaml

# Étapes B et C : modèles 2D, deux seeds, deux entraînements en parallèle sur CPU
for seed in 0 1; do
  $PY scripts/train.py --config configs/unet_2d.yaml --seed "$seed" --threads "$THREADS" &
  $PY scripts/train.py --config configs/wavelet_unet.yaml --seed "$seed" --threads "$THREADS" &
  wait
  $PY scripts/train.py --config configs/wavelet_unet_multiscale.yaml --seed "$seed" --threads $((2 * THREADS))
done

# Étape E : U-Net 3D
$PY scripts/train.py --config configs/unet_3d.yaml --seed 0 --threads $((2 * THREADS))

# Étapes C et D : évaluation sur le test. Seed 0 dans les 5 conditions (configs/evaluation.yaml),
# puis seed 1 et U-Net 3D sur les images originales et les facteurs 2 et 4.
$PY scripts/evaluate.py --config configs/evaluation.yaml
$PY scripts/evaluate.py --config configs/evaluation.yaml --conditions original down_x2 down_x4 \
  --runs experiments/unet2d_seed1 experiments/wavelet_unet_swt_haar_seed1 \
         experiments/wavelet_unet_dwt_haar_ms_seed1 experiments/unet3d_seed0
$PY scripts/analyze_results.py --split test

# Extension facultative : recalage
$PY scripts/registration_demo.py

"""Entraînement d'un modèle à partir d'un fichier de configuration.

Exemples :
    python scripts/train.py --config configs/unet_2d.yaml --seed 0
    python scripts/train.py --config configs/wavelet_unet.yaml --seed 0
    python scripts/train.py --config configs/unet_3d.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import train
from src.utils import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="remplace experiment.seed")
    parser.add_argument("--max-epochs", type=int, help="remplace training.max_epochs (tests rapides)")
    parser.add_argument("--samples-per-epoch", type=int, help="remplace training.samples_per_epoch")
    parser.add_argument("--name", help="nom du dossier de sortie (par défaut <nom>_seed<seed>)")
    parser.add_argument("--threads", type=int, help="nombre de threads CPU utilisés par PyTorch")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.samples_per_epoch:
        cfg["training"]["samples_per_epoch"] = args.samples_per_epoch
    if args.threads:
        cfg["training"]["num_threads"] = args.threads
    name = args.name or f"{cfg['experiment']['name']}_seed{cfg['experiment']['seed']}"
    run_dir = resolve(cfg["experiment"]["output_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "train.log", "a", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        if cfg["model"].get("type") == "unet3d":
            from src.training.trainer3d import train_3d
            summary = train_3d(cfg, run_dir, log)
        else:
            summary = train(cfg, run_dir, log)
        log(f"Terminé : {summary}")


if __name__ == "__main__":
    main()

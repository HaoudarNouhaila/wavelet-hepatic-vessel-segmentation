"""Évaluation des modèles entraînés, par volume, sur un ensemble donné.

Exemples :
    python scripts/evaluate.py --config configs/evaluation.yaml
    python scripts/evaluate.py --config configs/evaluation.yaml --conditions original
    python scripts/evaluate.py --config configs/evaluation.yaml --split val --runs experiments/unet2d_seed0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.data.splits import load_split
from src.evaluation.evaluator import evaluate, load_run
from src.utils import get_device, load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--runs", nargs="*", help="dossiers d'expériences (remplace la configuration)")
    parser.add_argument("--conditions", nargs="*", help="sous-ensemble des conditions à évaluer")
    parser.add_argument("--split", help="train, val ou test (remplace la configuration)")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-csv", help="fichier de sortie (remplace output_csv, utile en parallèle)")
    parser.add_argument("--threads", type=int, help="threads CPU utilisés par PyTorch")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    data_cfg = load_config(resolve(cfg["data_config"]))
    split_name = args.split or cfg["split"]
    case_ids = load_split(resolve(data_cfg["split"]["file"]), "subset")[split_name][: args.max_cases]
    conditions = cfg["conditions"]
    if args.conditions:
        conditions = {k: conditions[k] for k in args.conditions}
    run_dirs = [resolve(r) for r in (args.runs or cfg["runs"])]
    missing = [str(r) for r in run_dirs if not (r / "best.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoints absents : {missing}")

    torch.set_num_threads(args.threads or cfg.get("num_threads", torch.get_num_threads()))
    device = get_device(cfg.get("device", "auto"))
    runs = [load_run(r, device) for r in run_dirs]
    out_csv = resolve(args.output_csv or cfg["output_csv"].format(split=split_name))
    existing = pd.read_csv(out_csv) if out_csv.exists() else None
    pred_dir = resolve(cfg["predictions_dir"].format(split=split_name)) if cfg.get("save_predictions") else None

    print(f"{len(runs)} modèles, {len(case_ids)} volumes ({split_name}), conditions : {list(conditions)}")
    new = evaluate(runs, resolve(data_cfg["dataset"]["raw_dir"]), case_ids, conditions,
                   data_cfg["preprocessing"].get("target_inplane_spacing"), device, pred_dir, existing, out_csv)
    print(f"{len(new)} lignes ajoutées dans {out_csv}")


if __name__ == "__main__":
    main()

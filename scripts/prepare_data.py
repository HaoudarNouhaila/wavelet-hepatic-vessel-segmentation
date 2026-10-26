"""Prétraitement des cas du sous-ensemble et calcul des statistiques.

1. Statistiques d'intensité (voxels annotés des volumes d'entraînement).
2. Prétraitement de chaque cas (réorientation, normalisation, stockage .npy).
3. Statistiques de normalisation des sous-bandes ondelettes (entraînement).

Exemple :
    python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from src.data.msd import case_paths, load_case
from src.data.splits import load_split
from src.preprocessing.intensity import fit_intensity_stats, sample_values
from src.preprocessing.pipeline import load_processed, preprocess_case, save_processed
from src.utils import load_config, resolve
from src.wavelets.transforms import WaveletFeatures


def wavelet_key(spec: dict) -> str:
    levels = spec.get("levels", 1) if spec["mode"] == "dwt_multilevel" else 1
    return f"{spec['mode']}_{spec['wavelet']}_L{levels}"


def training_slices(processed_dir: Path, case_ids, per_case: int, seed: int):
    """Coupes d'entraînement utilisées pour estimer les statistiques ondelettes
    (moitié avec structures annotées, moitié tirées au hasard)."""
    rng = np.random.default_rng(seed)
    for cid in case_ids:
        case = load_processed(processed_dir / cid)
        z = case["image"].shape[0]
        fg = np.unique(case["fg_coords"][:, 0]) if len(case["fg_coords"]) else np.arange(z)
        chosen = list(rng.choice(fg, min(per_case // 2, len(fg)), replace=False))
        chosen += list(rng.choice(z, per_case - len(chosen), replace=False))
        for k in chosen:
            yield np.asarray(case["image"][int(k)], dtype=np.float32)


def foreground_values(raw_dir: Path, case_id: str) -> np.ndarray:
    case = load_case(raw_dir, case_id)
    return case.image[case.label > 0]


def preprocess_one(raw_dir: Path, case_id: str, split_name: str, stats: dict, target_inplane,
                   max_fg_coords: int, seed: int, out_dir: Path) -> str:
    case = load_case(raw_dir, case_id)
    processed = preprocess_case(case, stats, target_inplane, max_fg_coords, seed)
    processed["meta"]["split"] = split_name
    save_processed(out_dir, processed)
    return case_id


def run_parallel(function, jobs: list[tuple], workers: int, desc: str) -> list:
    """Exécute `function(*job)` pour chaque job, en conservant l'ordre des résultats."""
    if workers <= 1:
        return [function(*job) for job in tqdm(jobs, desc=desc)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(function, *zip(*jobs)), total=len(jobs), desc=desc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="processus parallèles (résultat identique)")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    pp = cfg["preprocessing"]
    raw_dir = resolve(cfg["dataset"]["raw_dir"])
    processed_dir = resolve(pp["processed_dir"])
    split = load_split(resolve(cfg["split"]["file"]), "subset")
    seed = cfg["split"]["seed"]

    stats_file = resolve(pp["stats_file"])
    if stats_file.exists() and not args.overwrite:
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
    else:
        # Lecture éventuellement parallèle, tirage séquentiel : même résultat quel que soit --workers.
        values = run_parallel(foreground_values, [(raw_dir, cid) for cid in split["train"]], args.workers,
                              "statistiques d'intensité")
        rng = np.random.default_rng(seed)
        samples = [sample_values(v, pp["max_fg_voxels_per_case"], rng) for v in values]
        stats = fit_intensity_stats(samples, *pp["intensity_percentiles"])
        stats["train_cases"] = list(split["train"])
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Normalisation :", {k: v for k, v in stats.items() if k != "train_cases"})

    all_ids = [(name, cid) for name in ("train", "val", "test") for cid in split[name]]
    missing, jobs = [], []
    for name, cid in all_ids:
        out = processed_dir / cid
        if (out / "meta.json").exists() and not args.overwrite:
            continue
        if not all(p.exists() for p in case_paths(raw_dir, cid)):
            missing.append(cid)  # pas encore téléchargé : traité lors d'une prochaine exécution
            continue
        jobs.append((raw_dir, cid, name, stats, pp.get("target_inplane_spacing"), pp["max_fg_coords"], seed, out))
    run_parallel(preprocess_one, jobs, args.workers, "prétraitement")
    if missing:
        print(f"{len(missing)} cas non encore téléchargés, à traiter plus tard : {missing}")

    wavelet_file = resolve(pp["wavelet_stats_file"])
    wstats = json.loads(wavelet_file.read_text(encoding="utf-8")) if wavelet_file.exists() and not args.overwrite else {}
    for spec in pp["wavelet_features"]:
        key = wavelet_key(spec)
        if key in wstats:
            continue
        extractor = WaveletFeatures(spec["mode"], spec["wavelet"], spec.get("levels", 1))
        slices = training_slices(processed_dir, split["train"], pp["wavelet_stats_slices_per_case"], seed)
        wstats[key] = extractor.fit(slices)
        print(f"Statistiques {key} :", json.dumps(wstats[key]))
    wavelet_file.write_text(json.dumps(wstats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

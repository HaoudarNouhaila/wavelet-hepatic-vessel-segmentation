"""Inspection du jeu de données Task08_HepaticVessel.

Produit :
- statistiques de géométrie de tous les cas (à partir des en-têtes NIfTI) ;
- statistiques d'intensité et de labels des cas téléchargés ;
- figures : distributions, histogrammes d'intensité, exemples de coupes.

Exemple :
    python scripts/inspect_dataset.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from src.data.msd import list_cases, load_case, read_dataset_json
from src.data.splits import load_split
from src.utils import load_config, resolve
from src.visualization.plots import BLUE, MUTED, ORANGE, RED, _imshow_ct, _overlay, label_legend, plot_case_views, save


def case_statistics(case) -> tuple[dict, dict]:
    img, lab = case.image, case.label
    voxel_ml = np.prod(case.spacing) / 1000
    vessel, tumour = lab == 1, lab == 2
    _, n_vessel_cc = ndimage.label(vessel, ndimage.generate_binary_structure(3, 3))
    row = {
        "case_id": case.case_id, "nz": img.shape[0], "ny": img.shape[1], "nx": img.shape[2],
        "sz_mm": case.spacing[0], "sy_mm": case.spacing[1], "sx_mm": case.spacing[2],
        "orientation": "".join(case.original_axcodes),
        "hu_min": float(img.min()), "hu_max": float(img.max()),
        "hu_p01": float(np.percentile(img, 1)), "hu_p99": float(np.percentile(img, 99)),
        "vessel_hu_mean": float(img[vessel].mean()) if vessel.any() else np.nan,
        "tumour_hu_mean": float(img[tumour].mean()) if tumour.any() else np.nan,
        "vessel_voxels": int(vessel.sum()), "tumour_voxels": int(tumour.sum()),
        "vessel_fraction": float(vessel.mean()), "tumour_fraction": float(tumour.mean()),
        "vessel_ml": float(vessel.sum() * voxel_ml), "tumour_ml": float(tumour.sum() * voxel_ml),
        "slices_with_vessel": int(vessel.any(axis=(1, 2)).sum()),
        "slices_with_tumour": int(tumour.any(axis=(1, 2)).sum()),
        "vessel_components": int(n_vessel_cc),
    }
    rng = np.random.default_rng(0)
    samples = {
        "tous les voxels": rng.choice(img.ravel(), 20000),
        "vaisseaux": rng.choice(img[vessel], min(5000, vessel.sum())) if vessel.any() else np.array([]),
        "tumeurs": rng.choice(img[tumour], min(5000, tumour.sum())) if tumour.any() else np.array([]),
    }
    return row, samples


def plot_geometry(headers: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    tr = headers[headers["subset"] == "imagesTr"]
    axes[0].hist(tr["nz"], bins=30, color=BLUE, edgecolor="white")
    axes[0].set(title="Nombre de coupes par volume", xlabel="coupes", ylabel="volumes")
    axes[1].hist(tr["sz_mm"], bins=30, color=BLUE, edgecolor="white")
    axes[1].set(title="Épaisseur de coupe", xlabel="espacement z (mm)", ylabel="volumes")
    axes[2].hist(tr["sx_mm"], bins=30, color=BLUE, edgecolor="white")
    axes[2].set(title="Espacement dans le plan", xlabel="espacement x = y (mm)", ylabel="volumes")
    fig.suptitle(f"Géométrie des {len(tr)} volumes annotés (en-têtes NIfTI)")
    return save(fig, path)


def plot_intensities(samples: dict[str, list], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(-1100, 600, 120)
    for name, color in (("tous les voxels", MUTED), ("vaisseaux", RED), ("tumeurs", ORANGE)):
        values = np.concatenate(samples[name])
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=name)
    ax.set(xlabel="Intensité (HU)", ylabel="densité", title="Distribution des intensités (cas téléchargés)")
    ax.legend()
    return save(fig, path)


def plot_label_stats(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(df["vessel_fraction"] * 100, bins=20, color=RED, edgecolor="white")
    axes[0].set(title="Part des voxels annotés vaisseaux", xlabel="% des voxels du volume", ylabel="volumes")
    axes[1].scatter(df["vessel_ml"], df["tumour_ml"], s=22, color=BLUE, edgecolors="white")
    axes[1].set(title="Volumes annotés par cas", xlabel="vaisseaux (mL)", ylabel="tumeurs (mL)")
    return save(fig, path)


def plot_gallery(cases: list, path: Path) -> Path:
    fig, axes = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 6.6))
    for col, case in enumerate(cases):
        z = int(np.argmax((case.label == 1).sum(axis=(1, 2))))
        _imshow_ct(axes[0, col], case.image[z])
        axes[0, col].set_title(f"{case.case_id}\ncoupe {z}", fontsize=8)
        _imshow_ct(axes[1, col], case.image[z])
        _overlay(axes[1, col], case.label[z])
    label_legend(axes[1, -1])
    fig.suptitle("Coupes axiales contenant le plus de voxels vaisseaux (haut : CT, bas : masque)")
    return save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    raw_dir = resolve(cfg["dataset"]["raw_dir"])
    fig_dir = resolve(cfg["inspection"]["figures_dir"])

    info = read_dataset_json(raw_dir)
    print("dataset.json :", {k: info[k] for k in ("name", "modality", "labels", "numTraining", "numTest") if k in info})
    print("Licence indiquée :", info.get("licence"), "| référence :", info.get("reference"))

    headers_csv = resolve(cfg["inspection"]["header_survey_csv"])
    if headers_csv.exists():
        headers = pd.read_csv(headers_csv)
        plot_geometry(headers, fig_dir / "geometrie_volumes.png")
        print(headers.groupby("subset")[["nx", "ny", "nz", "sx_mm", "sy_mm", "sz_mm"]].describe().T.round(3).to_string())
        print("Tailles dans le plan :", headers.groupby(["nx", "ny"]).size().to_dict())

    split = load_split(resolve(cfg["split"]["file"]), "subset")
    which = {cid: name for name, ids in split.items() for cid in ids}
    available = [c for c in list_cases(raw_dir) if c in which]
    rows, samples = [], {"tous les voxels": [], "vaisseaux": [], "tumeurs": []}
    gallery = []
    for cid in tqdm(available, desc="inspection"):
        case = load_case(raw_dir, cid)
        row, s = case_statistics(case)
        row["split"] = which[cid]
        rows.append(row)
        for k in samples:
            samples[k].append(s[k])
        if len(gallery) < 6:
            gallery.append(case)
        if len(rows) <= 2:
            plot_case_views(case.image, case.label, case.spacing, cid, fig_dir / f"vues_{cid}.png")
    df = pd.DataFrame(rows)
    out_csv = resolve(cfg["inspection"]["inspection_csv"])
    df.to_csv(out_csv, index=False)
    plot_intensities(samples, fig_dir / "histogrammes_intensites.png")
    plot_label_stats(df, fig_dir / "statistiques_labels.png")
    plot_gallery(gallery, fig_dir / "galerie_coupes.png")
    cols = ["nz", "sz_mm", "sx_mm", "hu_min", "hu_max", "vessel_hu_mean", "tumour_hu_mean", "vessel_fraction",
            "vessel_ml", "tumour_ml", "slices_with_vessel", "vessel_components"]
    print(df[cols].describe().T.round(4).to_string())
    print("Orientations d'origine :", df["orientation"].value_counts().to_dict())
    print(f"Inspection enregistrée dans {out_csv} et {fig_dir}")


if __name__ == "__main__":
    main()

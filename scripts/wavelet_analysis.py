"""Analyse des sous-bandes ondelettes sur des coupes CT réelles.

1. Visualisation des sous-bandes LL, LH, HL, HH (DWT Haar niveaux 1 et 2,
   SWT Haar, comparaison Haar / Daubechies 2).
2. Vérification de la reconstruction inverse (erreur maximale).
3. Amplitude moyenne des coefficients dans les vaisseaux et autour, sur
   les images d'origine et dégradées.

Exemple :
    python scripts/wavelet_analysis.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from src.data.msd import load_case
from src.data.splits import load_split
from src.preprocessing.degradation import degrade
from src.preprocessing.intensity import normalize_ct
from src.utils import load_config, resolve
from src.visualization.plots import BLUE, ORANGE, AQUA, MUTED, ct_window, crop_box, save
from src.wavelets.transforms import (SUBBANDS, dwt2_subbands, idwt2_reconstruct, iswt2_reconstruct, multilevel_dwt,
                                     swt2_subbands)

CONDITIONS = {"original": {}, "down_x2": {"downsample": 2}, "down_x4": {"downsample": 4}}
BAND_TITLES = {"LL": "LL (approximation)", "LH": "LH (détails horizontaux)", "HL": "HL (détails verticaux)",
               "HH": "HH (détails diagonaux)"}


def show_band(ax, band, title, detail: bool):
    if detail:
        vmax = np.percentile(np.abs(band), 99.5) or 1
        ax.imshow(band, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    else:
        ax.imshow(band, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def figure_subbands(image, label, case_id, z, out_dir: Path) -> None:
    box = crop_box(label > 0, margin=30, min_size=256)
    box = tuple(slice(s.start - s.start % 4, s.stop - s.stop % 4) for s in box)
    crop, lab = image[box], label[box]
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    axes[0, 0].imshow(ct_window(crop * 1.0), cmap="gray")
    axes[0, 0].contour(lab == 1, levels=[0.5], colors=[ORANGE], linewidths=0.7)
    axes[0, 0].set_title("CT (contour des vaisseaux)", fontsize=9)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([]); axes[0, 0].grid(False)
    rows = [("DWT Haar, niveau 1", multilevel_dwt(crop, 2, "haar")[0]),
            ("DWT Haar, niveau 2", multilevel_dwt(crop, 2, "haar")[1]),
            ("SWT Haar (non décimée)", swt2_subbands(crop, "haar"))]
    for r, (name, bands) in enumerate(rows):
        for c, key in enumerate(SUBBANDS, start=1):
            band = bands[key]
            show_band(axes[r, c], band, f"{name}\n{BAND_TITLES[key]} {band.shape[0]}x{band.shape[1]}", key != "LL")
    for r in (1, 2):
        axes[r, 0].axis("off")
    axes[1, 0].text(0.0, 0.6, "Les sous-bandes de détail sont affichées\navec une échelle symétrique\n"
                    "(bleu : négatif, rouge : positif).\nLa DWT divise la taille par 2 à chaque\n"
                    "niveau ; la SWT conserve la taille.", fontsize=9, color=MUTED, transform=axes[1, 0].transAxes)
    fig.suptitle(f"Décomposition en ondelettes d'une coupe CT ({case_id}, coupe {z}, HU)")
    save(fig, out_dir / "sous_bandes_haar.png")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for r, wavelet in enumerate(("haar", "db2")):
        bands = dwt2_subbands(crop, wavelet)
        for c, key in enumerate(SUBBANDS):
            show_band(axes[r, c], bands[key], f"{wavelet} - {BAND_TITLES[key]}", key != "LL")
    fig.suptitle("Comparaison Haar / Daubechies 2 (DWT niveau 1)")
    save(fig, out_dir / "haar_vs_db2.png")


def reconstruction_errors(slices: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for wavelet in ("haar", "db2", "db4"):
        for name, forward, inverse in (
            ("DWT", lambda x, w: dwt2_subbands(x, w), lambda b, w: idwt2_reconstruct(b, w)),
            ("SWT", lambda x, w: swt2_subbands(x, w), lambda b, w: iswt2_reconstruct(b, w)),
        ):
            errs = [np.abs(inverse(forward(s, wavelet), wavelet) - s).max() for s in slices]
            rows.append({"transform": name, "wavelet": wavelet, "max_abs_error_hu": float(np.max(errs)),
                         "n_slices": len(slices)})
    return pd.DataFrame(rows)


def band_energy(image_norm, label, region_px: int = 5) -> dict:
    """Amplitude moyenne |coefficient| (SWT Haar) dans les vaisseaux et dans un anneau autour."""
    vessel = label == 1
    ring = ndimage.binary_dilation(vessel, iterations=region_px) & ~vessel & (label == 0)
    bands = swt2_subbands(image_norm, "haar")
    out = {}
    for key in SUBBANDS:
        mag = np.abs(bands[key])
        out[f"{key}_vessel"] = float(mag[vessel].mean())
        out[f"{key}_ring"] = float(mag[ring].mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--cases", type=int, default=6, help="nombre de cas d'entraînement analysés")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    raw_dir = resolve(cfg["dataset"]["raw_dir"])
    stats = json.loads(resolve(cfg["preprocessing"]["stats_file"]).read_text(encoding="utf-8"))
    out_dir = resolve("results/figures/wavelets")
    case_ids = load_split(resolve(cfg["split"]["file"]), "subset")["train"][: args.cases]

    rows, recon_slices = [], []
    for i, cid in enumerate(case_ids):
        case = load_case(raw_dir, cid)
        z_list = np.flatnonzero((case.label == 1).sum(axis=(1, 2)) > 200)
        z_list = z_list[:: max(1, len(z_list) // 4)][:4]
        if i == 0:
            z0 = int(np.argmax((case.label == 1).sum(axis=(1, 2))))
            figure_subbands(case.image[z0].astype(np.float64), case.label[z0], cid, z0, out_dir)
        for cond_name, cond in CONDITIONS.items():
            degraded = degrade(case.image, case.spacing, cond, seed=i)
            norm = normalize_ct(degraded, stats)
            for z in z_list:
                rows.append({"case_id": cid, "slice": int(z), "condition": cond_name,
                             **band_energy(norm[z].astype(np.float64), case.label[z])})
        recon_slices += [case.image[z].astype(np.float64) for z in z_list[:2]]

    recon = reconstruction_errors(recon_slices)
    recon.to_csv(resolve("results/metrics/wavelet_reconstruction.csv"), index=False)
    print(recon.to_string(index=False))
    energy = pd.DataFrame(rows)
    energy.to_csv(resolve("results/metrics/wavelet_band_energy.csv"), index=False)
    summary = energy.groupby("condition").mean(numeric_only=True).reindex(list(CONDITIONS)).drop(columns="slice")
    print(summary.round(4).T.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(SUBBANDS))
    for ax, region, title in ((axes[0], "vessel", "Dans les vaisseaux"), (axes[1], "ring", "Anneau de 5 pixels autour")):
        for k, (cond, color) in enumerate(zip(CONDITIONS, (BLUE, ORANGE, AQUA))):
            values = [summary.loc[cond, f"{b}_{region}"] for b in SUBBANDS]
            ax.bar(x + (k - 1) * 0.25, values, width=0.22, color=color, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels(SUBBANDS)
        ax.set_yscale("log")
        ax.set(title=title, ylabel="|coefficient| moyen (CT normalisé)")
    axes[0].legend(title="condition")
    fig.suptitle("Amplitude des sous-bandes SWT Haar selon la résolution (coupes d'entraînement)")
    save(fig, out_dir / "energie_sous_bandes_degradations.png")


if __name__ == "__main__":
    main()

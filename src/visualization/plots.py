"""Figures du projet (Matplotlib).

Couleurs : les trois modèles utilisent les trois premières couleurs d'une
palette catégorielle vérifiée pour les daltonismes courants. Les cartes
d'erreurs utilisent les mêmes teintes (VP, FP, FN).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # scripts sans affichage ; les notebooks gardent l'affichage intégré
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE, AQUA, RED, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MODEL_LABELS = {
    "unet2d": "U-Net 2D",
    "wavelet_unet_swt_haar": "Wavelet U-Net (SWT Haar, entrée)",
    "wavelet_unet_swt_db2": "Wavelet U-Net (SWT db2, entrée)",
    "wavelet_unet_dwt_haar_ms": "Wavelet U-Net (DWT Haar, multi-échelle)",
    "unet3d": "U-Net 3D (MONAI)",
}
MODEL_SHORT = {
    "unet2d": "U-Net 2D",
    "wavelet_unet_swt_haar": "Wavelet\nSWT",
    "wavelet_unet_swt_db2": "Wavelet\nSWT db2",
    "wavelet_unet_dwt_haar_ms": "Wavelet\nmulti-éch.",
    "unet3d": "U-Net 3D",
}
MODEL_COLORS = {
    "unet2d": BLUE,
    "wavelet_unet_swt_haar": ORANGE,
    "wavelet_unet_dwt_haar_ms": AQUA,
    "wavelet_unet_swt_db2": "#6250d6",
    "unet3d": "#e87ba4",
}
LABEL_COLORS = {1: RED, 2: YELLOW}
ERROR_CMAP = ListedColormap([(0, 0, 0, 0), AQUA, ORANGE, BLUE])  # TN transparent, VP, FP, FN


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "legend.frameon": False,
    })


setup_style()


def ct_window(image_hu: np.ndarray, level: float = 80, width: float = 400) -> np.ndarray:
    """Fenêtrage CT pour l'affichage (fenêtre abdominale par défaut)."""
    lo, hi = level - width / 2, level + width / 2
    return np.clip((image_hu - lo) / (hi - lo), 0, 1)


def _imshow_ct(ax, image, aspect=1.0, windowed=True):
    ax.imshow(ct_window(image) if windowed else image, cmap="gray", aspect=aspect, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _overlay(ax, label, alpha=0.55, aspect=1.0):
    for value, color in LABEL_COLORS.items():
        mask = np.ma.masked_where(label != value, label)
        ax.imshow(mask, cmap=ListedColormap([color]), alpha=alpha, aspect=aspect, interpolation="nearest")


def label_legend(ax, loc="lower right"):
    ax.legend(handles=[Patch(color=RED, label="vaisseaux"), Patch(color=YELLOW, label="tumeurs")],
              loc=loc, fontsize=8, frameon=True, facecolor="white", framealpha=0.8)


def best_slices(label: np.ndarray, value: int = 1, n: int = 3) -> list[int]:
    """Indices de coupes réparties dans la zone contenant le label (au moins n coupes)."""
    counts = (label == value).sum(axis=(1, 2))
    idx = np.flatnonzero(counts)
    if len(idx) == 0:
        return list(np.linspace(0, label.shape[0] - 1, n).astype(int))
    order = idx[np.argsort(counts[idx])[::-1]]
    top = sorted(order[: max(n * 4, n)])
    return [top[int(i)] for i in np.linspace(0, len(top) - 1, n)]


def save(fig, path: Path | None) -> Path | None:
    """Enregistre et ferme la figure ; avec path=None, la figure est laissée ouverte (notebooks)."""
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_case_views(image_hu, label, spacing, case_id: str, path: Path) -> Path:
    """Coupe axiale, coronale et sagittale, sans et avec masque de référence."""
    sz, sh, sw = spacing
    z = best_slices(label, 1, 1)[0]
    ys, xs = np.nonzero(label[z] == 1) if (label[z] == 1).any() else (np.array([image_hu.shape[1] // 2]),) * 2
    y, x = int(np.median(ys)), int(np.median(xs))
    views = [
        ("axiale", image_hu[z], label[z], sh / sw),
        ("coronale", image_hu[::-1, y, :], label[::-1, y, :], sz / sw),
        ("sagittale", image_hu[::-1, :, x], label[::-1, :, x], sz / sh),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for col, (name, img, lab, aspect) in enumerate(views):
        _imshow_ct(axes[0, col], img, aspect)
        axes[0, col].set_title(f"Coupe {name}")
        _imshow_ct(axes[1, col], img, aspect)
        _overlay(axes[1, col], lab, aspect=aspect)
        axes[1, col].set_title(f"Coupe {name} + masque")
    label_legend(axes[1, 2])
    fig.suptitle(f"{case_id} - espacement (z, y, x) = ({sz:.2f}, {sh:.2f}, {sw:.2f}) mm, fenêtre [-120, 280] HU")
    return save(fig, path)


def plot_prediction_preview(image_hu, pred, path: Path, title: str = "", label=None) -> Path:
    slices = best_slices(pred if label is None else label, 1, 3)
    rows = 1 if label is None else 2
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4.2 * rows), squeeze=False)
    for col, z in enumerate(slices):
        _imshow_ct(axes[0, col], image_hu[z])
        _overlay(axes[0, col], pred[z])
        axes[0, col].set_title(f"Prédiction, coupe {z}")
        if label is not None:
            _imshow_ct(axes[1, col], image_hu[z])
            _overlay(axes[1, col], label[z])
            axes[1, col].set_title(f"Référence, coupe {z}")
    label_legend(axes[0, 2])
    fig.suptitle(title)
    return save(fig, path)


save_prediction_preview = plot_prediction_preview


def plot_training_curves(histories: dict[str, pd.DataFrame], path: Path, title: str = "") -> Path:
    """Pertes d'entraînement / validation et Dice de validation, un trait par modèle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for model, runs in histories.items():
        color = MODEL_COLORS.get(model, MUTED)
        label = MODEL_LABELS.get(model, model)
        for i, df in enumerate(runs if isinstance(runs, list) else [runs]):
            kw = {"color": color, "alpha": 1.0 if i == 0 else 0.45, "label": label if i == 0 else None}
            axes[0].plot(df["epoch"], df["train_loss"], **kw)
            axes[1].plot(df["epoch"], df["val_loss"], **kw)
            axes[2].plot(df["epoch"], df["val_dice_vessel"], **kw)
    axes[0].set_title("Perte d'entraînement (Dice + CE)")
    axes[1].set_title("Perte de validation (volumes complets)")
    axes[2].set_title("Dice vaisseaux en validation")
    for ax in axes:
        ax.set_xlabel("Époque")
    axes[0].set_ylabel("Perte")
    axes[2].set_ylabel("Dice")
    axes[2].legend(loc="lower right", fontsize=8)
    if title:
        fig.suptitle(title)
    return save(fig, path)


def plot_metric_by_model(df: pd.DataFrame, metrics: list[tuple[str, str]], path: Path, title: str = "",
                         models: list[str] | None = None) -> Path:
    """Distribution par volume (points) et moyenne (trait) de plusieurs métriques, par modèle."""
    models = models or [m for m in MODEL_LABELS if m in set(df["model"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (col, name) in zip(axes[0], metrics):
        for i, model in enumerate(models):
            values = df.loc[df["model"] == model, col].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=18, color=MODEL_COLORS[model],
                       alpha=0.6, edgecolors="white", linewidths=0.6)
            if len(values):
                ax.hlines(np.mean(values), i - 0.3, i + 0.3, color=INK, linewidth=2)
                ax.text(i + 0.32, np.mean(values), f"{np.mean(values):.3f}", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=8)
        ax.set_title(name)
        ax.set_xlim(-0.5, len(models) - 0.3)
    if title:
        fig.suptitle(title)
    return save(fig, path)


def plot_degradation(summary: pd.DataFrame, metrics: list[tuple[str, str]], conditions: list[str],
                     condition_labels: dict[str, str], path: Path, title: str = "") -> Path:
    """Moyenne ± écart-type (sur les volumes) de chaque métrique selon la condition.

    `summary` est indexé par (model, condition) avec des colonnes (métrique, "mean" | "std").
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2), squeeze=False)
    x = np.arange(len(conditions))
    present = set(summary.index.get_level_values("model"))
    models = [m for m in MODEL_LABELS if m in present]
    for ax, (col, name) in zip(axes[0], metrics):
        for k, model in enumerate(models):
            sub = summary.xs(model, level="model").reindex(conditions)
            offset = (k - (len(models) - 1) / 2) * 0.08
            ax.errorbar(x + offset, sub[(col, "mean")], yerr=sub[(col, "std")], color=MODEL_COLORS[model],
                        marker="o", markersize=5, capsize=3, label=MODEL_LABELS[model])
        ax.set_xticks(x)
        ax.set_xticklabels([condition_labels.get(c, c) for c in conditions], rotation=30, ha="right", fontsize=8)
        ax.set_title(name)
    fig.legend(handles=model_handles(models), loc="lower center", ncol=len(models), fontsize=8,
               bbox_to_anchor=(0.5, -0.13))
    if title:
        fig.suptitle(title)
    return save(fig, path)


SEED_STYLES = {0: {"linestyle": "-", "marker": "o"}, 1: {"linestyle": "--", "marker": "s"},
               2: {"linestyle": ":", "marker": "^"}}


def plot_degradation_by_seed(per_seed: pd.DataFrame, metrics: list[tuple[str, str]], conditions: list[str],
                             condition_labels: dict[str, str], path: Path, title: str = "") -> Path:
    """Moyenne sur les volumes de chaque métrique selon la condition, un trait par modèle et par seed.

    `per_seed` contient les colonnes model, seed, condition et les métriques (une ligne par run).
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.4), squeeze=False)
    x = {c: i for i, c in enumerate(conditions)}
    models = [m for m in MODEL_LABELS if m in set(per_seed["model"])]
    for ax, (col, name) in zip(axes[0], metrics):
        for k, model in enumerate(models):
            for seed, sub in per_seed[per_seed["model"] == model].groupby("seed"):
                sub = sub[sub["condition"].isin(conditions)].sort_values("condition", key=lambda s: s.map(x))
                offset = (k - (len(models) - 1) / 2) * 0.06
                ax.plot(sub["condition"].map(x) + offset, sub[col], color=MODEL_COLORS[model], markersize=5,
                        alpha=1.0 if seed == 0 else 0.75, **SEED_STYLES.get(seed, {}))
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([condition_labels.get(c, c) for c in conditions], rotation=30, ha="right", fontsize=8)
        ax.set_title(name)
    handles = model_handles(models) + [Line2D([0], [0], color=MUTED, label=f"seed {s}", **SEED_STYLES[s])
                                       for s in sorted(per_seed["seed"].unique())]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.13))
    if title:
        fig.suptitle(title)
    return save(fig, path)


def plot_seed_comparison(per_seed: pd.DataFrame, metrics: list[tuple[str, str]], path: Path, title: str = "") -> Path:
    """Valeur moyenne sur les volumes pour chaque seed (points) et moyenne des seeds (trait), par modèle."""
    models = [m for m in MODEL_LABELS if m in set(per_seed["model"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.2), squeeze=False)
    for ax, (col, name) in zip(axes[0], metrics):
        for i, model in enumerate(models):
            sub = per_seed[per_seed["model"] == model]
            for _, row in sub.iterrows():
                ax.scatter(i, row[col], color=MODEL_COLORS[model], s=40, edgecolors="white", linewidths=1,
                           marker=SEED_STYLES.get(int(row["seed"]), {}).get("marker", "o"), zorder=3)
            ax.hlines(sub[col].mean(), i - 0.25, i + 0.25, color=INK, linewidth=2)
            ax.text(i + 0.28, sub[col].mean(), f"{sub[col].mean():.3f}", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=8)
        ax.set_xlim(-0.5, len(models) - 0.2)
        ax.set_title(name)
    handles = [Line2D([0], [0], color=MUTED, linestyle="none", label=f"seed {s}", **{"marker": SEED_STYLES[s]["marker"]})
               for s in sorted(per_seed["seed"].unique())]
    axes[0][-1].legend(handles=handles + [Line2D([0], [0], color=INK, label="moyenne des seeds")], fontsize=8)
    if title:
        fig.suptitle(title)
    return save(fig, path)


def crop_box(mask: np.ndarray, margin: int = 20, min_size: int = 160) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return slice(None), slice(None)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max((ys.max() - ys.min()) // 2 + margin, (xs.max() - xs.min()) // 2 + margin, min_size // 2)
    h, w = mask.shape
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    return slice(y0, min(y0 + 2 * half, h)), slice(x0, min(x0 + 2 * half, w))


def plot_model_comparison_slice(image_hu, label, preds: dict[str, np.ndarray], z: int, path: Path,
                                title: str = "", zoom: bool = True) -> Path:
    """Image, référence, puis pour chaque modèle prédiction et carte d'erreurs (vaisseaux)."""
    box = crop_box(label[z] > 0) if zoom else (slice(None), slice(None))
    n = len(preds)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.6 * (n + 1), 7.4))
    img = image_hu[z][box]
    _imshow_ct(axes[0, 0], img)
    axes[0, 0].set_title("CT")
    _imshow_ct(axes[1, 0], img)
    _overlay(axes[1, 0], label[z][box])
    axes[1, 0].set_title("Référence")
    for col, (model, pred) in enumerate(preds.items(), start=1):
        _imshow_ct(axes[0, col], img)
        _overlay(axes[0, col], pred[z][box])
        axes[0, col].set_title(MODEL_LABELS.get(model, model), fontsize=9)
        _imshow_ct(axes[1, col], img)
        err = np.zeros(img.shape, dtype=np.uint8)
        p, g = pred[z][box] == 1, label[z][box] == 1
        err[p & g], err[p & ~g], err[~p & g] = 1, 2, 3
        axes[1, col].imshow(err, cmap=ERROR_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[1, col].set_title("Erreurs (vaisseaux)", fontsize=9)
    axes[1, -1].legend(handles=[Patch(color=AQUA, label="vrai positif"), Patch(color=ORANGE, label="faux positif"),
                                Patch(color=BLUE, label="faux négatif")], loc="lower right", fontsize=7,
                       frameon=True, facecolor="white", framealpha=0.85)
    label_legend(axes[1, 0])
    fig.suptitle(f"{title} (coupe {z})")
    return save(fig, path)


def plot_vessels_3d(mask: np.ndarray, spacing, path: Path, title: str = "", step: int = 2, other=None) -> Path:
    """Rendu surfacique (marching cubes) d'un ou deux masques de vaisseaux."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    masks = [("référence" if other is not None else "", mask, RED)]
    if other is not None:
        masks.append(("prédiction", other, BLUE))
    meshes = []
    for _, m, _ in masks:
        if m.any():
            verts, faces, _, _ = marching_cubes(m.astype(np.float32), 0.5, spacing=tuple(spacing), step_size=step)
            meshes.append(verts[:, [2, 1, 0]][faces])  # axes affichés : x (gauche-droite), y, z vertical
        else:
            meshes.append(None)
    # mêmes bornes pour les deux vues, calées sur la référence
    ref = meshes[0].reshape(-1, 3) if meshes[0] is not None else np.zeros((1, 3))
    lo, hi = ref.min(0), ref.max(0)
    fig = plt.figure(figsize=(6 * len(masks), 6))
    for i, ((name, _, color), tris) in enumerate(zip(masks, meshes), start=1):
        ax = fig.add_subplot(1, len(masks), i, projection="3d")
        if tris is not None:
            ax.add_collection3d(Poly3DCollection(tris, alpha=0.9, facecolor=color, edgecolor="none"))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1))
        ax.view_init(elev=20, azim=-65)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title(name)
    fig.suptitle(title)
    return save(fig, path)


def model_handles(models) -> list[Line2D]:
    return [Line2D([0], [0], color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in models]

"""Recalage sur transformations synthétiques connues (extension facultative).

Pour un volume CT, plusieurs transformations de référence (translation,
rigide, affine) sont appliquées, puis estimées par recalage. L'erreur est
mesurée sur les paramètres et sur des points tirés dans les vaisseaux (TRE).

Exemple :
    python scripts/registration_demo.py --case hepaticvessel_XXX
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.data.msd import load_case
from src.data.splits import load_split
from src.registration.synthetic import (image_center, make_transform, register, resample, sample_points,
                                        target_registration_error, volume_to_sitk)
from src.utils import load_config, resolve
from src.visualization.plots import BLUE, ORANGE, AQUA, ct_window, save

SCENARIOS = {
    "translation": {"kind": "translation", "translation_mm": (6.0, -4.0, 5.0)},
    "rigide": {"kind": "rigid", "rotation_deg": (2.0, -3.0, 6.0), "translation_mm": (5.0, 3.0, -4.0)},
    "affine": {"kind": "affine", "rotation_deg": (1.0, 2.0, -4.0), "translation_mm": (-3.0, 4.0, 2.0),
               "scale": (1.04, 0.97, 1.02)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--case", help="identifiant du cas (par défaut : premier cas de validation)")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--threads", type=int, default=0, help="threads SimpleITK (0 : valeur par défaut)")
    args = parser.parse_args()
    if args.threads:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    cfg = load_config(resolve(args.config))
    case_id = args.case or load_split(resolve(cfg["split"]["file"]), "subset")["val"][0]
    case = load_case(resolve(cfg["dataset"]["raw_dir"]), case_id)
    fixed = volume_to_sitk(np.clip(case.image, -1024, 2000), case.spacing)
    center = image_center(fixed)
    points = sample_points(fixed, case.label == 1, args.points)

    rows, examples = [], {}
    for name, params in SCENARIOS.items():
        reference = make_transform(center=center, **params)
        moving = resample(fixed, reference)
        estimated, info = register(fixed, moving, params["kind"])
        # registration : fixed(x) ~ moving(T(x)) et moving(x) = fixed(T_gen(x)), donc T doit approcher T_gen^-1
        truth = reference.GetInverse()
        tre = target_registration_error(estimated, truth, points)
        identity = sitk.TranslationTransform(3)
        tre_before = target_registration_error(identity, truth, points)
        rows.append({"case_id": case_id, "scenario": name, "transform": params["kind"],
                     "tre_before_mean_mm": tre_before.mean(), "tre_mean_mm": tre.mean(),
                     "tre_median_mm": np.median(tre), "tre_max_mm": tre.max(),
                     "iterations": info["iterations"], "final_metric": info["final_metric"], "time_s": info["time_s"],
                     "stop_condition": info["stop"]})
        print(rows[-1])
        examples[name] = (moving, resample(moving, estimated), info["metric_history"])

    out = resolve("results/metrics/registration_synthetic.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    z = int(np.argmax((case.label == 1).sum(axis=(1, 2))))
    fixed_arr = sitk.GetArrayFromImage(fixed)
    fig, axes = plt.subplots(len(examples), 3, figsize=(11, 3.6 * len(examples)))
    for row, (name, (moving, registered, _)) in enumerate(examples.items()):
        mov = sitk.GetArrayFromImage(moving)
        reg = sitk.GetArrayFromImage(registered)
        axes[row, 0].imshow(ct_window(fixed_arr[z]), cmap="gray")
        axes[row, 0].set_title(f"{name} : image fixe (coupe {z})", fontsize=9)
        axes[row, 1].imshow(np.abs(ct_window(mov[z]) - ct_window(fixed_arr[z])), cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_title("|fixe - mobile| avant recalage", fontsize=9)
        axes[row, 2].imshow(np.abs(ct_window(reg[z]) - ct_window(fixed_arr[z])), cmap="magma", vmin=0, vmax=1)
        axes[row, 2].set_title("|fixe - mobile recalée| après recalage", fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"Recalage sur transformations synthétiques connues ({case_id}, même volume)")
    save(fig, resolve("results/figures/registration/differences_avant_apres.png"))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, (_, _, history)), color in zip(examples.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(history, color=color, label=name)
    ax.set(xlabel="itération (tous niveaux de résolution)", ylabel="information mutuelle de Mattes (à minimiser)",
           title="Convergence de l'optimisation")
    ax.legend()
    save(fig, resolve("results/figures/registration/convergence.png"))
    print(f"Résultats : {out}")


if __name__ == "__main__":
    main()

"""Segmentation d'un volume CT NIfTI avec un modèle entraîné.

Le masque prédit (0 fond, 1 vaisseaux, 2 tumeurs) est enregistré en NIfTI
dans l'orientation RAS+, avec l'affine correspondante : il se superpose au
CT d'origine dans un visualiseur (ITK-SNAP, 3D Slicer).

Exemple :
    python scripts/predict.py --run experiments/unet2d_seed0 \
        --input data/raw/Task08_HepaticVessel/imagesTr/hepaticvessel_001.nii.gz \
        --output experiments/predictions/hepaticvessel_001_pred.nii.gz --png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from src.data.msd import load_nifti_zhw
from src.evaluation.evaluator import load_run, prepare_volume
from src.evaluation.inference import predict_volume, probs_to_labels, probs_to_original_grid
from src.utils import get_device, resolve


def zhw_to_ras(array: np.ndarray) -> np.ndarray:
    """Inverse de src.data.msd.to_zhw."""
    return np.ascontiguousarray(np.flip(array.transpose(2, 1, 0), axis=(0, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="dossier d'expérience contenant best.pt")
    parser.add_argument("--input", required=True, help="volume CT .nii.gz")
    parser.add_argument("--output", required=True, help="masque prédit .nii.gz")
    parser.add_argument("--png", action="store_true", help="enregistrer aussi un aperçu PNG")
    args = parser.parse_args()

    device = get_device()
    run = load_run(resolve(args.run), device)
    image, spacing, affine, _ = load_nifti_zhw(Path(args.input))
    target = run["cfg"]["data"].get("target_inplane_spacing")
    volume, _ = prepare_volume(image, spacing, run["stats"], {}, target, seed=0)
    probs = predict_volume(run["model"], volume, run["wavelet"], run["fill"], device=device)
    pred = probs_to_labels(probs_to_original_grid(probs, image.shape))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(zhw_to_ras(pred), affine), str(out))
    print(f"Masque enregistré : {out} ; voxels vaisseaux : {(pred == 1).sum()}, tumeurs : {(pred == 2).sum()}")

    if args.png:
        from src.visualization.plots import save_prediction_preview
        png = out.with_name(out.name.replace(".nii.gz", ".png"))
        save_prediction_preview(image, pred, png, title=f"{Path(args.input).name} - {run['name']}")
        print(f"Aperçu : {png}")


if __name__ == "__main__":
    main()

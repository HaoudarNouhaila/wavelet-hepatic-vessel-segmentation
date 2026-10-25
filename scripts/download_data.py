"""Téléchargement de Task08_HepaticVessel (complet ou sous-ensemble).

Étapes :
1. lecture de l'index de l'archive tar officielle (en-têtes uniquement) ;
2. création de la séparation train / val / test au niveau patient ;
3. lecture des en-têtes NIfTI de tous les cas (dimensions, espacement),
   sans télécharger les volumes ;
4. téléchargement des volumes et masques du sous-ensemble choisi.

Exemple :
    python scripts/download_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from src.data.msd import case_paths, list_cases
from src.data.remote_tar import RangeReader, build_index, download_member, load_index, save_index
from src.data.splits import make_split, save_split, select_subset
from src.utils import load_config, resolve

PREFIX = "Task08_HepaticVessel/"


def case_members(members) -> dict[str, dict]:
    """Associe chaque cas à ses fichiers image et masque dans l'archive."""
    cases: dict[str, dict] = {}
    for m in members:
        name = m.name[2:] if m.name.startswith("./") else m.name
        parts = name.split("/")
        if len(parts) != 3 or parts[2].startswith(".") or not parts[2].endswith(".nii.gz"):
            continue
        folder, case_id = parts[1], parts[2][: -len(".nii.gz")]
        cases.setdefault(case_id, {})[folder] = m
    return cases


def remote_header(reader: RangeReader, member, nbytes: int = 8192) -> nib.Nifti1Header:
    """Décompresse le début d'un .nii.gz distant pour lire son en-tête."""
    raw = reader.read(member.offset, min(nbytes, member.size))
    data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    return nib.Nifti1Header.from_fileobj(io.BytesIO(data[:348]))


def survey_headers(reader, cases: dict, out_csv: Path) -> pd.DataFrame:
    rows = []
    for case_id in tqdm(sorted(cases), desc="en-têtes NIfTI"):
        for folder in ("imagesTr", "imagesTs"):
            if folder in cases[case_id]:
                hdr = remote_header(reader, cases[case_id][folder])
                shape = hdr.get_data_shape()
                zooms = hdr.get_zooms()
                rows.append({
                    "case_id": case_id, "subset": folder, "nx": shape[0], "ny": shape[1], "nz": shape[2],
                    "sx_mm": float(zooms[0]), "sy_mm": float(zooms[1]), "sz_mm": float(zooms[2]),
                    "dtype": str(hdr.get_data_dtype()), "file_mb": cases[case_id][folder].size / 2**20,
                    "has_label": "labelsTr" in cases[case_id],
                })
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-survey", action="store_true", help="ne pas relire les en-têtes NIfTI distants")
    parser.add_argument("--dry-run", action="store_true", help="afficher la sélection sans télécharger")
    parser.add_argument("--from-local", action="store_true",
                        help="archive déjà extraite dans raw_dir : créer seulement la séparation")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    ds, sp = cfg["dataset"], cfg["split"]

    if args.from_local:
        raw_dir = resolve(ds["raw_dir"])
        labelled = [c for c in list_cases(raw_dir) if case_paths(raw_dir, c)[1].exists()]
        split = make_split(labelled, sp["seed"], tuple(sp["fractions"]))
        subset = select_subset(split, sp["subset_sizes"])
        save_split(resolve(sp["file"]), split, subset, sp["seed"], sp["fractions"])
        print(f"{len(labelled)} cas annotés trouvés ; sous-ensemble : {', '.join(f'{k}={len(v)}' for k, v in subset.items())}")
        return

    reader = RangeReader(ds["url"])

    index_file = resolve(ds["index_file"])
    if index_file.exists():
        members = load_index(index_file)
    else:
        print("Lecture des en-têtes de l'archive (une requête par fichier)...")
        members = build_index(reader)
        save_index(members, index_file)
    cases = case_members(members)
    labelled = sorted(c for c, f in cases.items() if "imagesTr" in f and "labelsTr" in f)
    print(f"{len(members)} entrées dans l'archive, {len(labelled)} cas annotés, "
          f"{sum('imagesTs' in f for f in cases.values())} cas de test non annotés")

    raw_dir = resolve(ds["raw_dir"])
    dataset_json = next(m for m in members if m.name.endswith("dataset.json") and "/._" not in m.name)
    download_member(ds["url"], dataset_json, raw_dir / PREFIX / "dataset.json")

    if not args.skip_survey:
        survey = survey_headers(reader, cases, resolve(cfg["inspection"]["header_survey_csv"]))
        print(survey.groupby("subset")[["nx", "ny", "nz", "sx_mm", "sz_mm"]].describe().T.round(3))

    split = make_split(labelled, sp["seed"], tuple(sp["fractions"]))
    subset = select_subset(split, sp["subset_sizes"])
    save_split(resolve(sp["file"]), split, subset, sp["seed"], sp["fractions"])
    wanted = [c for name in ("train", "val", "test") for c in subset[name]]
    files = [cases[c][f] for c in wanted for f in ("imagesTr", "labelsTr")]
    total = sum(m.size for m in files) / 2**30
    print(f"Sous-ensemble : {', '.join(f'{k}={len(v)}' for k, v in subset.items())} ; {total:.2f} Go à télécharger")
    if args.dry_run:
        return
    for m in tqdm(files, desc="téléchargement", unit="fichier"):
        rel = m.name[2:] if m.name.startswith("./") else m.name
        download_member(ds["url"], m, raw_dir / rel)
        if rel.endswith(".nii.gz"):
            with gzip.open(raw_dir / rel, "rb") as handle:  # contrôle d'intégrité du flux gzip
                while handle.read(1 << 24):
                    pass


if __name__ == "__main__":
    main()

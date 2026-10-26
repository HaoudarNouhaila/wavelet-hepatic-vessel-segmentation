"""Séparation entraînement / validation / test au niveau patient.

Dans Task08, chaque volume correspond à un cas distinct (hepaticvessel_XXX).
La séparation est faite sur les identifiants de cas, jamais sur les coupes,
ce qui garantit qu'aucune coupe d'un même volume n'apparaît dans deux
ensembles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


def make_split(
    case_ids: list[str],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Mélange les cas avec une seed fixe et les répartit en trois ensembles."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("Les fractions doivent sommer à 1.")
    ids = sorted(set(case_ids))
    if len(ids) != len(case_ids):
        raise ValueError("Identifiants de cas dupliqués.")
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def select_subset(split: dict[str, list[str]], sizes: dict[str, int]) -> dict[str, list[str]]:
    """Garde les premiers cas de chaque ensemble (ordre déjà aléatoire).

    Utile quand les ressources ne permettent pas d'utiliser tous les cas :
    le sous-ensemble reste contenu dans la séparation complète.
    """
    return {name: split[name][: sizes[name]] for name in SPLIT_NAMES}


def check_disjoint(split: dict[str, list[str]]) -> None:
    """Lève une erreur si un cas appartient à plusieurs ensembles."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for case in split.get(name, []):
            if case in seen:
                raise ValueError(f"{case} présent dans {seen[case]} et {name}")
            seen[case] = name


def save_split(path: Path, split: dict, subset: dict, seed: int, fractions) -> None:
    check_disjoint(split)
    check_disjoint(subset)
    payload = {
        "seed": seed,
        "fractions": list(fractions),
        "full": split,
        "subset": subset,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path, which: str = "subset") -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = payload[which]
    check_disjoint(split)
    return split

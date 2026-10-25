"""Décompositions en ondelettes 2D avec PyWavelets.

Nommage des sous-bandes (niveau 1) par rapport aux sorties de pywt.dwt2,
qui retourne (cA, (cH, cV, cD)) :
- LL = cA : approximation (basse fréquence dans les deux directions) ;
- LH = cH : détails horizontaux (contours horizontaux) ;
- HL = cV : détails verticaux (contours verticaux) ;
- HH = cD : détails diagonaux.

Ces transformées sont calculées en prétraitement (NumPy), en dehors du
graphe PyTorch : elles ne sont pas différentiables et ne sont pas apprises.
Elles ne sont appliquées qu'aux images CT, jamais aux masques.
"""

from __future__ import annotations

import numpy as np
import pywt

SUBBANDS = ("LL", "LH", "HL", "HH")


def dwt2_subbands(image: np.ndarray, wavelet: str = "haar", mode: str = "periodization") -> dict[str, np.ndarray]:
    """DWT 2D à un niveau (sous-bandes de taille H/2 x W/2).

    Le mode "periodization" donne exactement H/2 x W/2 pour des tailles
    paires. Avec Haar, aucun coefficient ne dépend du bord de l'image.
    """
    cA, (cH, cV, cD) = pywt.dwt2(image, wavelet, mode=mode)
    return {"LL": cA, "LH": cH, "HL": cV, "HH": cD}


def idwt2_reconstruct(subbands: dict[str, np.ndarray], wavelet: str = "haar", mode: str = "periodization") -> np.ndarray:
    """Transformée inverse à partir des quatre sous-bandes."""
    coeffs = (subbands["LL"], (subbands["LH"], subbands["HL"], subbands["HH"]))
    return pywt.idwt2(coeffs, wavelet, mode=mode)


def multilevel_dwt(image: np.ndarray, levels: int = 2, wavelet: str = "haar", mode: str = "periodization") -> list[dict[str, np.ndarray]]:
    """DWT itérée sur l'approximation : la liste contient un dict par niveau.

    Le niveau k a une taille H/2^k x W/2^k, ce qui correspond à la résolution
    de l'étage k de l'encodeur d'un U-Net.
    """
    out = []
    current = image
    for _ in range(levels):
        bands = dwt2_subbands(current, wavelet, mode)
        out.append(bands)
        current = bands["LL"]
    return out


def swt2_subbands(image: np.ndarray, wavelet: str = "haar", level: int = 1) -> dict[str, np.ndarray]:
    """Transformée stationnaire (non décimée) : sous-bandes de même taille que l'image.

    Chaque coefficient est calculé à toutes les positions, sans
    sous-échantillonnage, ce qui évite de devoir ré-échantillonner les
    sous-bandes pour les aligner sur l'image. Les dimensions doivent être
    divisibles par 2**level.
    """
    coeffs = pywt.swt2(image, wavelet, level=level, start_level=0, trim_approx=False, norm=False)
    cA, (cH, cV, cD) = coeffs[0]  # niveau le plus grossier en premier
    return {"LL": cA, "LH": cH, "HL": cV, "HH": cD}


def iswt2_reconstruct(subbands: dict[str, np.ndarray], wavelet: str = "haar") -> np.ndarray:
    coeffs = [(subbands["LL"], (subbands["LH"], subbands["HL"], subbands["HH"]))]
    return pywt.iswt2(coeffs, wavelet, norm=False)


class WaveletFeatures:
    """Calcule les caractéristiques ondelettes d'une coupe CT prétraitée.

    Deux modes sont proposés :
    - "swt" : 4 sous-bandes pleine résolution (canaux d'entrée supplémentaires) ;
    - "dwt_multilevel" : sous-bandes décimées des niveaux 1..L, destinées à
      être injectées dans l'encodeur à la résolution correspondante.

    Chaque sous-bande est standardisée avec une moyenne et un écart-type
    estimés sur l'ensemble d'entraînement uniquement (voir `fit`).
    """

    def __init__(self, mode: str = "swt", wavelet: str = "haar", levels: int = 1, stats: dict | None = None):
        if mode not in ("swt", "dwt_multilevel"):
            raise ValueError(f"Mode inconnu : {mode}")
        self.mode = mode
        self.wavelet = wavelet
        self.levels = levels if mode == "dwt_multilevel" else 1
        self.stats = stats

    @property
    def multiple(self) -> int:
        """Les côtés de l'image doivent être multiples de cette valeur."""
        return 2 ** self.levels

    def raw(self, image: np.ndarray) -> list[np.ndarray]:
        """Retourne une liste de tableaux (4, h, w), un par niveau."""
        if image.shape[0] % self.multiple or image.shape[1] % self.multiple:
            raise ValueError(f"Taille {image.shape} non divisible par {self.multiple}")
        if self.mode == "swt":
            bands = swt2_subbands(image, self.wavelet, level=1)
            return [np.stack([bands[k] for k in SUBBANDS])]
        levels = multilevel_dwt(image, self.levels, self.wavelet)
        return [np.stack([lvl[k] for k in SUBBANDS]) for lvl in levels]

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        if self.stats is None:
            raise RuntimeError("Statistiques de normalisation absentes : appeler fit() d'abord.")
        out = []
        for level, arr in enumerate(self.raw(image.astype(np.float32))):
            mean = np.asarray(self.stats["mean"][level], dtype=np.float32)[:, None, None]
            std = np.asarray(self.stats["std"][level], dtype=np.float32)[:, None, None]
            out.append(((arr - mean) / std).astype(np.float32))
        return out

    def fit(self, images) -> dict:
        """Estime moyenne et écart-type par sous-bande et par niveau."""
        sums = sq = counts = None
        for image in images:
            raw = self.raw(image.astype(np.float64))
            if sums is None:
                sums = [np.zeros(4) for _ in raw]
                sq = [np.zeros(4) for _ in raw]
                counts = [0 for _ in raw]
            for level, arr in enumerate(raw):
                flat = arr.reshape(4, -1)
                sums[level] += flat.sum(axis=1)
                sq[level] += (flat ** 2).sum(axis=1)
                counts[level] += flat.shape[1]
        if sums is None:
            raise ValueError("Aucune image pour estimer les statistiques.")
        stats = {"mean": [], "std": []}
        for level in range(len(sums)):
            mean = sums[level] / counts[level]
            var = np.maximum(sq[level] / counts[level] - mean ** 2, 1e-12)
            stats["mean"].append(mean.tolist())
            stats["std"].append(np.sqrt(var).tolist())
        self.stats = stats
        return stats

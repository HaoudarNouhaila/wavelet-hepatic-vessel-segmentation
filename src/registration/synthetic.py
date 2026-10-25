"""Introduction au recalage avec SimpleITK sur des transformations synthétiques.

Principe : une image "mobile" est fabriquée en appliquant une transformation
connue T_gen au CT d'origine, puis on recale l'original (image fixe) sur
cette image mobile. Comme T_gen est connue, l'erreur peut être mesurée
exactement (erreur de recalage sur des points cibles, TRE).

Il s'agit d'un exercice de validation sur un même volume, pas d'un
recalage entre deux acquisitions ni entre modalités (par exemple CT /
micro-CT).
"""

from __future__ import annotations

import time

import numpy as np
import SimpleITK as sitk


def volume_to_sitk(volume_zhw: np.ndarray, spacing_zhw) -> sitk.Image:
    """Tableau (Z, H, W) -> image SimpleITK (axes x = W, y = H, z = Z)."""
    image = sitk.GetImageFromArray(volume_zhw.astype(np.float32))
    image.SetSpacing(tuple(float(s) for s in spacing_zhw[::-1]))
    return image


def image_center(image: sitk.Image) -> tuple[float, float, float]:
    size = np.array(image.GetSize()) - 1
    return image.TransformContinuousIndexToPhysicalPoint((size / 2).tolist())


def make_transform(kind: str, center, rotation_deg=(0, 0, 0), translation_mm=(0, 0, 0), scale=(1, 1, 1)):
    """Transformation de référence : translation, rigide (Euler 3D) ou affine."""
    if kind == "translation":
        return sitk.TranslationTransform(3, tuple(float(t) for t in translation_mm))
    rot = sitk.Euler3DTransform()
    rot.SetCenter(center)
    rot.SetRotation(*np.deg2rad(rotation_deg))
    rot.SetTranslation(tuple(float(t) for t in translation_mm))
    if kind == "rigid":
        return rot
    affine = sitk.AffineTransform(3)
    affine.SetCenter(center)
    matrix = np.array(rot.GetMatrix()).reshape(3, 3) @ np.diag(scale)
    affine.SetMatrix(matrix.ravel().tolist())
    affine.SetTranslation(rot.GetTranslation())
    return affine


def resample(image: sitk.Image, transform, default: float = -1024.0, interpolator=sitk.sitkLinear) -> sitk.Image:
    """Image transformée : out(x) = image(T(x))."""
    return sitk.Resample(image, image, transform, interpolator, default, sitk.sitkFloat32)


def register(fixed: sitk.Image, moving: sitk.Image, kind: str, seed: int = 0) -> tuple[sitk.Transform, dict]:
    """Recalage multi-résolution par information mutuelle de Mattes.

    Retourne la transformation estimée T telle que fixed(x) ~ moving(T(x)).
    """
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=300,
                                                 relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    center = image_center(fixed)
    if kind == "translation":
        initial = sitk.TranslationTransform(3)
    elif kind == "rigid":
        initial = sitk.Euler3DTransform()
        initial.SetCenter(center)
    else:
        initial = sitk.AffineTransform(3)
        initial.SetCenter(center)
    reg.SetInitialTransform(initial, inPlace=False)
    history = []
    reg.AddCommand(sitk.sitkIterationEvent, lambda: history.append(reg.GetMetricValue()))
    t0 = time.time()
    final = reg.Execute(fixed, moving)
    info = {
        "iterations": reg.GetOptimizerIteration(), "final_metric": reg.GetMetricValue(),
        "stop": reg.GetOptimizerStopConditionDescription(), "time_s": time.time() - t0, "metric_history": history,
    }
    return final, info


def target_registration_error(estimated, reference, points) -> np.ndarray:
    """Distance (mm) entre les images des points par la transformation estimée et la référence."""
    return np.array([np.linalg.norm(np.subtract(estimated.TransformPoint(p), reference.TransformPoint(p)))
                     for p in points])


def sample_points(image: sitk.Image, mask_zhw: np.ndarray, n: int, seed: int = 0) -> list[tuple]:
    """Points physiques tirés dans un masque (par exemple les vaisseaux)."""
    idx = np.argwhere(mask_zhw)
    rng = np.random.default_rng(seed)
    chosen = idx[rng.choice(len(idx), min(n, len(idx)), replace=False)]
    return [image.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in chosen]

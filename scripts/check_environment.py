"""Vérification de l'environnement : Python, bibliothèques, GPU, mémoire, disque.

Exemple :
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys

PACKAGES = ["torch", "monai", "numpy", "scipy", "pandas", "sklearn", "skimage", "pywt", "nibabel", "SimpleITK",
            "matplotlib", "seaborn", "yaml", "pytest"]


def main() -> None:
    print(f"Système : {platform.platform()}")
    print(f"Python : {sys.version.split()[0]}")
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None) or module.Version_VersionString()
            print(f"  {name:<11} {version}")
        except Exception as exc:  # bibliothèque absente ou mal installée
            print(f"  {name:<11} ERREUR : {exc}")
    import torch

    print(f"CUDA disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU : {props.name}, {props.total_memory / 2**30:.1f} Go, CUDA {torch.version.cuda}")
    print(f"Threads CPU utilisés par PyTorch : {torch.get_num_threads()} (coeurs logiques : {os.cpu_count()})")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM : {mem.total / 2**30:.1f} Go au total, {mem.available / 2**30:.1f} Go disponibles")
    except ImportError:
        pass
    print(f"Espace disque libre : {shutil.disk_usage('.').free / 2**30:.1f} Go")


if __name__ == "__main__":
    main()

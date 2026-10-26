"""Vérifie que tout le code source est bien versionné (aucune règle .gitignore ne l'exclut)."""

import shutil
import subprocess

import pytest

from src.utils import PROJECT_ROOT


def test_all_sources_are_tracked_or_addable():
    if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
        pytest.skip("dépôt Git absent")
    sources = [p for folder in ("src", "scripts", "tests", "configs") for p in (PROJECT_ROOT / folder).rglob("*")
               if p.is_file() and p.suffix in {".py", ".yaml", ".sh"} and "__pycache__" not in p.parts]
    rel = [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in sources]
    result = subprocess.run(["git", "check-ignore", "--no-index", *rel], cwd=PROJECT_ROOT, capture_output=True, text=True)
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert ignored == [], f"fichiers source ignorés par .gitignore : {ignored}"

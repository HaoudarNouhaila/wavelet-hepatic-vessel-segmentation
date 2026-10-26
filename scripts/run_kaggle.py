"""Enchaînement des expériences sur GPU (Kaggle) avec un budget de temps global.

Étapes (--steps) :
  data      séparation au niveau patient à partir des fichiers extraits, puis prétraitement ;
  train     modèles 2D (3 modèles x seeds, seeds dans la boucle externe), puis U-Net 3D si le
            temps le permet ;
  evaluate  évaluation sur le test, une condition par processus ;
  analyze   fusion des résultats, tableaux, tests statistiques et figures.

Chaque étape reprend là où elle s'est arrêtée : les entraînements terminés et les volumes déjà
évalués sont sautés. Le budget est compté depuis le premier lancement (fichier .start_time dans
le dossier des expériences), pour que plusieurs cellules du notebook partagent la même limite.

Exemple (Kaggle) :
    python scripts/run_kaggle.py --steps data train evaluate analyze
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

MODELS_2D = [
    ("configs/unet_2d.yaml", "unet2d"),
    ("configs/wavelet_unet.yaml", "wavelet_unet_swt_haar"),
    ("configs/wavelet_unet_multiscale.yaml", "wavelet_unet_dwt_haar_ms"),
]
# Conditions évaluées pour tous les modèles, et conditions évaluées pour la seed 0 seulement
# (même protocole que sur CPU).
CONDITIONS_ALL = ["original", "down_x2", "down_x4"]
CONDITIONS_SEED0 = ["blur_1mm", "down_x2_noise20"]


class Runner:
    def __init__(self, args):
        self.args = args
        self.exp_dir = ROOT / args.experiments_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        start_file = self.exp_dir / ".start_time"
        if not start_file.exists():
            start_file.write_text(str(time.time()))
        self.start = float(start_file.read_text())

    def elapsed_h(self) -> float:
        return (time.time() - self.start) / 3600

    def log(self, message: str) -> None:
        line = f"[{self.elapsed_h():5.2f} h] {message}"
        print(line, flush=True)
        with open(self.exp_dir / "run_kaggle.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def run(self, cmd: list[str]) -> int:
        self.log("$ " + " ".join(cmd))
        return subprocess.run([sys.executable] + cmd, cwd=ROOT).returncode

    # Étapes --------------------------------------------------------------------------------

    def data(self) -> None:
        a = self.args
        self.run(["scripts/download_data.py", "--config", a.data_config, "--from-local"])
        if self.run(["scripts/prepare_data.py", "--config", a.data_config, "--workers", str(a.workers)]):
            raise RuntimeError("Échec du prétraitement")

    def train(self) -> None:
        a = self.args
        overrides_2d = [a.override_2d] + a.extra_override
        for seed in a.seeds:
            for config, name in MODELS_2D:
                self._train_one(config, overrides_2d, name, seed, a.train_budget_hours)
        if not a.no_3d:
            overrides_3d = [a.override_3d] + a.extra_override_3d
            remaining = a.train_budget_hours + a.extra_3d_hours - self.elapsed_h()
            if remaining > 0.75:
                self._train_one("configs/unet_3d.yaml", overrides_3d, "unet3d", 0,
                                a.train_budget_hours + a.extra_3d_hours)
            else:
                self.log("U-Net 3D non lancé : temps insuffisant")

    def _train_one(self, config: str, overrides: list[str], name: str, seed: int, budget_h: float) -> None:
        run_dir = self.exp_dir / f"{name}_seed{seed}"
        if (run_dir / "summary.json").exists():
            self.log(f"{run_dir.name} déjà terminé")
            return
        remaining = budget_h - self.elapsed_h()
        if remaining < self.args.min_run_hours:
            self.log(f"{run_dir.name} non lancé : il reste {remaining:.2f} h de budget d'entraînement")
            return
        code = self.run(["scripts/train.py", "--config", config, "--override", *overrides,
                         "--seed", str(seed), "--max-minutes", f"{remaining * 60:.0f}"])
        self.log(f"{run_dir.name} terminé (code {code})")

    def evaluate(self) -> None:
        a = self.args
        runs = sorted(p.parent for p in self.exp_dir.glob("*_seed*/best.pt"))
        if not runs:
            self.log("Aucun modèle entraîné à évaluer")
            return
        seed0 = [r for r in runs if r.name.endswith("_seed0")]
        parts_dir = ROOT / a.results_dir / "metrics" / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(cond, runs) for cond in CONDITIONS_ALL] + [(cond, seed0) for cond in CONDITIONS_SEED0]
        processes = []
        pending = list(jobs)
        while pending or processes:
            while pending and len(processes) < a.eval_processes:
                cond, run_list = pending.pop(0)
                cmd = [sys.executable, "scripts/evaluate.py", "--config", a.eval_config, "--conditions", cond,
                       "--threads", "1", "--output-csv", str(parts_dir / f"test_{cond}.csv"),
                       "--runs", *[str(r) for r in run_list]]
                if a.max_eval_cases:
                    cmd += ["--max-cases", str(a.max_eval_cases)]
                log_file = open(parts_dir / f"eval_{cond}.log", "a", encoding="utf-8")
                self.log(f"évaluation {cond} : {len(run_list)} modèles")
                processes.append((cond, subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT),
                                  log_file))
            time.sleep(a.poll_seconds)
            for item in list(processes):
                cond, proc, log_file = item
                if proc.poll() is not None:
                    log_file.close()
                    processes.remove(item)
                    self.log(f"évaluation {cond} terminée (code {proc.returncode})")
            if self.elapsed_h() > a.budget_hours - a.analysis_margin_hours:
                for cond, proc, log_file in processes:
                    proc.terminate()
                    log_file.close()
                    self.log(f"évaluation {cond} interrompue (budget) ; les volumes déjà évalués sont conservés")
                processes, pending = [], []

    def analyze(self) -> None:
        a = self.args
        metrics_dir = ROOT / a.results_dir / "metrics"
        parts = sorted((metrics_dir / "parts").glob("test_*.csv"))
        if not parts:
            self.log("Aucun résultat d'évaluation à analyser")
            return
        table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        table.to_csv(metrics_dir / "test_per_volume.csv", index=False)
        self.log(f"{len(table)} lignes fusionnées dans {metrics_dir / 'test_per_volume.csv'}")
        self.run(["scripts/analyze_results.py", "--split", "test", "--eval-config", a.eval_config,
                  "--results-dir", a.results_dir, "--experiments-dir", a.experiments_dir])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=["data", "train", "evaluate", "analyze"],
                        choices=["data", "train", "evaluate", "analyze"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget-hours", type=float, default=11.0, help="durée totale avant la fin de session")
    parser.add_argument("--train-budget-hours", type=float, default=6.5, help="fin des entraînements 2D")
    parser.add_argument("--extra-3d-hours", type=float, default=1.0, help="temps supplémentaire pour le U-Net 3D")
    parser.add_argument("--min-run-hours", type=float, default=0.3, help="ne pas lancer un run avec moins de temps")
    parser.add_argument("--analysis-margin-hours", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-processes", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-eval-cases", type=int)
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--data-config", default="configs/kaggle/data.yaml")
    parser.add_argument("--override-2d", default="configs/kaggle/gpu_2d.yaml")
    parser.add_argument("--override-3d", default="configs/kaggle/gpu_3d.yaml")
    parser.add_argument("--extra-override", nargs="*", default=[], help="YAML ajoutés après --override-2d")
    parser.add_argument("--extra-override-3d", nargs="*", default=[], help="YAML ajoutés après --override-3d")
    parser.add_argument("--eval-config", default="configs/kaggle/evaluation.yaml")
    parser.add_argument("--experiments-dir", default="experiments/kaggle")
    parser.add_argument("--results-dir", default="results/kaggle")
    args = parser.parse_args()

    runner = Runner(args)
    runner.log(f"étapes : {args.steps}, seeds : {args.seeds}")
    for step in args.steps:
        getattr(runner, step)()
    runner.log("fin")


if __name__ == "__main__":
    main()

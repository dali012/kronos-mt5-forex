"""Generate kaggle/kronos_walkforward.ipynb (valid nbformat v4 JSON).

Run: python kaggle/_make_notebook.py
Keeping the notebook as generated-from-source avoids hand-editing JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kronos_walkforward.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    }


cells = [
    md(
        "# Kronos FX — GPU walk-forward (Kaggle)",
        "",
        "Runs the Kronos + NautilusTrader walk-forward validation on a **GPU** so it",
        "finishes in minutes instead of ~1h on CPU.",
        "",
        "## One-time setup",
        "1. **Settings panel (right):** turn **Internet = ON** and **Accelerator = GPU T4 ×1** (or P100).",
        "2. **Add Data → Your Datasets:** attach the dataset that holds this repo.",
        "   - Create it by zipping the project folder (include `src/`, `backtest/`, `config/`,",
        "     `data/*.csv`, `pyproject.toml`; you can skip `.venv/` and `vendor/`).",
        "   - Note its mount path under `/kaggle/input/<your-slug>` and set `PROJECT_INPUT` below.",
        "3. **Run all.** Outputs (tearsheets + results CSV) land in `/kaggle/working` — download from the",
        "   Output tab.",
    ),
    md("## 1. Config — edit these"),
    code(
        "# Path where your attached dataset is mounted (adjust the slug to match yours):",
        "PROJECT_INPUT = '/kaggle/input/kronos-mt5-forex'",
        "",
        "SYMBOL_CSV = 'EURUSD_15M.csv'   # which pair to validate",
        "DEVICE = 'cuda:0'               # GPU; use 'cpu' only to debug",
        "",
        "# Walk-forward params (GPU is fast, so we can afford a finer cadence + more samples)",
        "TRAIN_DAYS = 90",
        "TEST_DAYS = 60",
        "STEP_DAYS = 60",
        "LOOKBACK = 256",
        "FORECAST_EVERY = 8     # forecast every N bars (8 = ~hourly on M15)",
        "SAMPLE_COUNT = 10      # sampled paths per forecast (cheap on GPU; sharpens prob_up)",
        "CALIBRATE = False      # True = re-fit the signal threshold on each train window",
        "",
        "# Alternative to a dataset: clone from GitHub instead (leave REPO_URL empty to use the dataset)",
        "REPO_URL = ''          # e.g. 'https://github.com/youruser/kronos-mt5-forex.git'",
    ),
    md("## 2. Stage the project + Kronos source into a writable dir"),
    code(
        "import os, shutil, subprocess, sys, pathlib",
        "",
        "WORK = pathlib.Path('/kaggle/working/proj')",
        "if WORK.exists():",
        "    shutil.rmtree(WORK)",
        "",
        "if REPO_URL:",
        "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(WORK)], check=True)",
        "else:",
        "    src = pathlib.Path(PROJECT_INPUT)",
        "    assert src.exists(), f'{src} not found — attach the dataset and fix PROJECT_INPUT'",
        "    # the dataset may be the repo root, or a single nested folder; resolve it",
        "    if not (src / 'pyproject.toml').exists():",
        "        cands = [p for p in src.iterdir() if (p / 'pyproject.toml').exists()]",
        "        assert cands, f'no pyproject.toml under {src}'",
        "        src = cands[0]",
        "    shutil.copytree(src, WORK)",
        "",
        "print('project at', WORK)",
        "",
        "# Vendored Kronos model code (cloned fresh; the brain adds it to sys.path automatically)",
        "KRONOS = WORK / 'vendor' / 'Kronos'",
        "if not (KRONOS / 'model').exists():",
        "    KRONOS.parent.mkdir(parents=True, exist_ok=True)",
        "    subprocess.run(['git', 'clone', '--depth', '1',",
        "                    'https://github.com/shiyu-coder/Kronos.git', str(KRONOS)], check=True)",
        "print('kronos at', KRONOS)",
    ),
    md("## 3. Install dependencies (torch + CUDA are preinstalled on Kaggle)"),
    code(
        "%pip install -q 'nautilus_trader==1.228.0' einops safetensors 'huggingface_hub>=0.33' \\",
        "    quantstats 'pydantic>=2' 'pydantic-settings>=2'",
        "",
        "import torch",
        "print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(),",
        "      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')",
    ),
    md("## 4. Sanity check — locate the CSV"),
    code(
        "data_csv = WORK / 'data' / SYMBOL_CSV",
        "if not data_csv.exists():",
        "    # data may have been uploaded separately at the dataset root",
        "    alt = pathlib.Path(PROJECT_INPUT) / SYMBOL_CSV",
        "    assert alt.exists(), f'{SYMBOL_CSV} not found in {data_csv} or {alt}'",
        "    (WORK / 'data').mkdir(exist_ok=True)",
        "    shutil.copy(alt, data_csv)",
        "import pandas as pd",
        "_df = pd.read_csv(data_csv)",
        "print(SYMBOL_CSV, len(_df), 'bars', _df['timestamps'].iloc[0], '->', _df['timestamps'].iloc[-1])",
    ),
    md("## 5. Run the walk-forward on GPU"),
    code(
        "env = dict(os.environ, PYTHONPATH='src')",
        "cmd = [sys.executable, '-m', 'backtest.walk_forward',",
        "       f'data/{SYMBOL_CSV}',",
        "       '--device', DEVICE,",
        "       '--train-days', str(TRAIN_DAYS),",
        "       '--test-days', str(TEST_DAYS),",
        "       '--step-days', str(STEP_DAYS),",
        "       '--lookback', str(LOOKBACK),",
        "       '--forecast-every', str(FORECAST_EVERY),",
        "       '--sample-count', str(SAMPLE_COUNT)]",
        "if not CALIBRATE:",
        "    cmd.append('--no-calibrate')",
        "print(' '.join(cmd))",
        "",
        "proc = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE,",
        "                        stderr=subprocess.STDOUT, text=True, bufsize=1)",
        "for line in proc.stdout:",
        "    print(line, end='')",
        "proc.wait()",
        "print('\\nexit code:', proc.returncode)",
    ),
    md("## 6. Collect outputs to /kaggle/working (downloadable from the Output tab)"),
    code(
        "import glob",
        "for html in glob.glob(str(WORK / 'reports' / '*.html')):",
        "    dst = pathlib.Path('/kaggle/working') / pathlib.Path(html).name",
        "    shutil.copy(html, dst)",
        "    print('saved', dst)",
        "",
        "from IPython.display import FileLink, display",
        "for html in glob.glob('/kaggle/working/*.html'):",
        "    display(FileLink(html))",
    ),
    md(
        "### To validate the other pairs",
        "Re-run from step 4 with `SYMBOL_CSV = 'GBPUSD_15M.csv'` or `'USDJPY_15M.csv'`.",
        "USDJPY exercises the JPY-quote position-sizing path (value/unit = 1/price).",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)

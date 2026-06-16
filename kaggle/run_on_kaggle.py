"""Drive the whole Kaggle GPU walk-forward from this workspace via the Kaggle CLI.

It packages the repo (code + CSVs) as a Kaggle Dataset, pushes a GPU+internet
script kernel that runs the walk-forward, polls until it finishes, and downloads
the tearsheet + metrics JSON back into reports/kaggle/.

Prereqs:
  - `pip install kaggle` (done in the project venv)
  - Kaggle API token at ~/.kaggle/kaggle.json (chmod 600). Get it from
    https://www.kaggle.com/settings -> "Create New Token".

Usage:
  python kaggle/run_on_kaggle.py --symbol EURUSD_15M.csv
  python kaggle/run_on_kaggle.py --symbol USDJPY_15M.csv --forecast-every 8 --sample-count 10
  python kaggle/run_on_kaggle.py --push-only      # upload + start, don't wait
  python kaggle/run_on_kaggle.py --fetch-only      # just download latest output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO / "kaggle" / "kernel"
STAGING = REPO / "kaggle" / "_staging"
OUT_DIR = REPO / "reports" / "kaggle"

DATASET_SLUG = "kronos-mt5-forex"
KERNEL_SLUG = "kronos-fx-walkforward"

# What goes into the dataset (code + data). Keep it lean: no venv / vendor / git.
INCLUDE = ["src", "backtest", "config", "data", "pyproject.toml"]


def _kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, check=check, text=True, capture_output=capture, cwd=str(REPO))


def _hf_token_from_dotenv() -> str | None:
    env = REPO / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _username() -> str:
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        raise SystemExit(
            "Kaggle token not found at ~/.kaggle/kaggle.json.\n"
            "Create one at https://www.kaggle.com/settings (Create New Token), then:\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        )
    return json.loads(token.read_text())["username"]


def stage_dataset(user: str, params: dict) -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for item in INCLUDE:
        src = REPO / item
        if src.is_dir():
            shutil.copytree(src, STAGING / item)
        elif src.exists():
            shutil.copy(src, STAGING / item)
    (STAGING / "kaggle_params.json").write_text(json.dumps(params, indent=2))
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Kronos MT5 Forex",
                "id": f"{user}/{DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
    )
    print(f"staged dataset at {STAGING} ({len(INCLUDE)} top-level items + params)")


def push_dataset(user: str) -> None:
    exists = (
        _kaggle(
            "datasets", "status", f"{user}/{DATASET_SLUG}", check=False, capture=True
        ).returncode
        == 0
    )
    if exists:
        print("dataset exists -> pushing a new version")
        _kaggle(
            "datasets",
            "version",
            "-p",
            str(STAGING),
            "-m",
            f"update {time.strftime('%Y-%m-%d %H:%M')}",
            "--dir-mode",
            "zip",
        )
    else:
        print("creating dataset")
        _kaggle("datasets", "create", "-p", str(STAGING), "--dir-mode", "zip")


def write_kernel_metadata(user: str) -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                # Title must slugify to KERNEL_SLUG, or Kaggle creates the kernel
                # under a different slug than `id` and status polling breaks.
                "id": f"{user}/{KERNEL_SLUG}",
                "title": "Kronos FX Walkforward",
                "code_file": "kronos_kaggle.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{user}/{DATASET_SLUG}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
    )


def wait_for_dataset(user: str, poll: int = 10, timeout: int = 600) -> None:
    """Datasets process asynchronously after upload; wait until ready so the
    kernel can see its input."""
    slug = f"{user}/{DATASET_SLUG}"
    start = time.time()
    while True:
        res = _kaggle("datasets", "status", slug, check=False, capture=True)
        out = ((res.stdout or "") + (res.stderr or "")).strip().lower()
        if "ready" in out:
            print("dataset ready")
            return
        if time.time() - start > timeout:
            print(
                f"WARNING: dataset status still '{out or 'unknown'}' after {timeout}s; proceeding"
            )
            return
        print(f"[dataset {int(time.time() - start)}s] {out or 'processing'}")
        time.sleep(poll)


def push_kernel() -> None:
    print("pushing kernel (GPU + internet)...")
    _kaggle("kernels", "push", "-p", str(KERNEL_DIR))


def wait_for_kernel(user: str, poll: int = 30, timeout: int = 5400) -> str:
    slug = f"{user}/{KERNEL_SLUG}"
    start = time.time()
    while True:
        res = _kaggle("kernels", "status", slug, check=False, capture=True)
        out = (res.stdout or "") + (res.stderr or "")
        status = out.strip().splitlines()[-1] if out.strip() else "(no status)"
        print(f"[{int(time.time() - start)}s] {status}")
        low = out.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "cancel" in low:
            return "error"
        if time.time() - start > timeout:
            return "timeout"
        time.sleep(poll)


def fetch_output(user: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(OUT_DIR))
    print(f"\nDownloaded kernel output to {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        print("  ", f.name)

    for fe in sorted(OUT_DIR.glob("forecast_eval*.json")):
        data = json.loads(fe.read_text())
        cfg = data.get("config", {})
        tag = cfg.get("resample") or "native"
        print(f"\n=== FORECAST QUALITY [{fe.name} | tf={tag} | model={cfg.get('model')}] ===")
        for k, v in data.get("stats", {}).items():
            print(f"  {k:32s}: {v}")

    results = OUT_DIR / "walkforward_results.json"
    if results.exists():
        data = json.loads(results.read_text())
        print("\n=== AGGREGATE OOS METRICS ===")
        for k, v in data.get("aggregate", {}).items():
            print(f"  {k:18s}: {v}")
        print("\n=== MONTE CARLO ===")
        for k, v in data.get("monte_carlo", {}).items():
            print(f"  {k:22s}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Kronos walk-forward on Kaggle GPU")
    ap.add_argument("--symbol", default="EURUSD_15M.csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--step-days", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--forecast-every", type=int, default=16)
    ap.add_argument("--sample-count", type=int, default=10)
    ap.add_argument("--task", choices=["forecast_eval", "walkforward", "both"], default="both")
    ap.add_argument("--eval-stride", type=int, default=16)
    ap.add_argument("--pred-len", type=int, default=12)
    ap.add_argument("--threshold-bps", type=float, default=8.0)
    ap.add_argument("--model", default="NeoQuasar/Kronos-small",
                    help="NeoQuasar/Kronos-small | NeoQuasar/Kronos-base | NeoQuasar/Kronos-mini")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base",
                    help="Kronos-mini needs NeoQuasar/Kronos-Tokenizer-2k")
    ap.add_argument("--use-volume", action="store_true",
                    help="feed real volume to Kronos (crypto/equities; for forecast_eval)")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="round-trip cost in bps for forecast_eval (relative; use for crypto)")
    ap.add_argument("--resample", default=None,
                    help="comma list of timeframes for forecast_eval, e.g. '1h,4h,1D'")
    ap.add_argument("--target-points", type=int, default=None,
                    help="auto-stride each timeframe to ~this many forecasts")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--push-only", action="store_true", help="upload + start, don't wait")
    ap.add_argument("--fetch-only", action="store_true", help="just download latest output")
    ap.add_argument(
        "--skip-dataset",
        action="store_true",
        help="don't re-upload the dataset; just (re)push + run the kernel",
    )
    args = ap.parse_args()

    user = _username()

    if args.fetch_only:
        fetch_output(user)
        return

    params = {
        "symbol_csv": args.symbol,
        "device": args.device,
        "task": args.task,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "lookback": args.lookback,
        "pred_len": args.pred_len,
        "forecast_every": args.forecast_every,
        "eval_stride": args.eval_stride,
        "sample_count": args.sample_count,
        "threshold_bps": args.threshold_bps,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "use_volume": args.use_volume,
        "cost_bps": args.cost_bps,
        "resample_list": [s.strip() for s in args.resample.split(",")] if args.resample else None,
        "target_points": args.target_points,
        "calibrate": args.calibrate,
    }
    # Optional HF token (faster/authenticated model downloads). Read from the
    # local env or the gitignored .env file; it travels only into the PRIVATE
    # Kaggle dataset, never the repo.
    hf_token = os.environ.get("HF_TOKEN") or _hf_token_from_dotenv()
    if hf_token:
        params["hf_token"] = hf_token
        print("HF_TOKEN found in env -> will authenticate Hugging Face downloads")
    else:
        print("no HF_TOKEN in env -> Hugging Face downloads stay unauthenticated (fine, just slower)")
    if not args.skip_dataset:
        stage_dataset(user, params)
        push_dataset(user)
        wait_for_dataset(user)
    write_kernel_metadata(user)
    push_kernel()

    if args.push_only:
        print(f"\nKernel started. Watch it at https://www.kaggle.com/code/{user}/{KERNEL_SLUG}")
        print("Fetch results later with:  python kaggle/run_on_kaggle.py --fetch-only")
        return

    status = wait_for_kernel(user)
    print(f"\nkernel finished: {status}")
    if status in ("complete", "error"):
        fetch_output(user)


if __name__ == "__main__":
    main()

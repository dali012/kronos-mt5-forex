"""Kaggle GPU kernel entry point (script-type kernel).

Pushed and run via `kaggle/run_on_kaggle.py`. On Kaggle it:
  1. reads run params from the attached dataset's `kaggle_params.json`,
  2. copies the project (also in the dataset) into the writable /kaggle/working,
  3. clones the Kronos model code + installs deps,
  4. runs the walk-forward on the GPU,
  5. leaves the tearsheet + metrics JSON in /kaggle/working (the kernel output).

Everything written to /kaggle/working is retrievable with
`kaggle kernels output <user>/<slug>`.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

WORKING = "/kaggle/working"
BUILD = "/tmp/proj"  # keep the project copy off /kaggle/working so output stays clean


def _ensure_torch_supports_gpu() -> None:
    """Kaggle may assign a P100 (sm_60); if the preinstalled torch was built
    without that arch, reinstall a cu121 wheel that covers sm_60..sm_90."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if not torch.cuda.is_available():
        return
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch = torch.cuda.get_arch_list()
    print(
        f"GPU {torch.cuda.get_device_name(0)} {sm}; torch {torch.__version__} archs {arch}",
        flush=True,
    )
    if sm in arch:
        return
    print(f"{sm} unsupported by current torch -> reinstalling torch 2.4.1 (cu121)", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--force-reinstall",
            "--no-deps",
            "torch==2.4.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print('reinstalled torch', torch.__version__, torch.cuda.get_arch_list())",
        ],
        capture_output=True,
        text=True,
    )
    print(check.stdout, check.stderr, flush=True)


def _find_input() -> str:
    # The dataset may mount at /kaggle/input/<slug> or, for private datasets,
    # /kaggle/input/datasets/<owner>/<slug> — so search recursively for a marker.
    for marker in ("kaggle_params.json", "pyproject.toml"):
        hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    print("DEBUG: /kaggle/input tree:", flush=True)
    for path in sorted(glob.glob("/kaggle/input/**", recursive=True))[:40]:
        print("  ", path, flush=True)
    raise SystemExit("No input dataset with the project found under /kaggle/input/")


def _resolve_project_root(base: str) -> str:
    if os.path.exists(os.path.join(base, "pyproject.toml")):
        return base
    for p in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(p, "pyproject.toml")):
            return p
    raise SystemExit(f"No pyproject.toml found under {base}")


def main() -> None:
    inp = _find_input()
    params_path = os.path.join(inp, "kaggle_params.json")
    params = json.load(open(params_path)) if os.path.exists(params_path) else {}
    print("PARAMS:", json.dumps(params), flush=True)

    project_src = _resolve_project_root(inp)
    work = BUILD
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(project_src, work)
    print("project ->", work, flush=True)

    # Kronos model code (the brain adds vendor/Kronos to sys.path automatically)
    kronos = os.path.join(work, "vendor", "Kronos")
    if not os.path.exists(os.path.join(kronos, "model")):
        os.makedirs(os.path.dirname(kronos), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/shiyu-coder/Kronos.git", kronos],
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "nautilus_trader==1.228.0",
            "einops",
            "safetensors",
            "huggingface_hub>=0.33",
            "quantstats",
            "pydantic>=2",
            "pydantic-settings>=2",
        ],
        check=True,
    )

    _ensure_torch_supports_gpu()

    csv = params.get("symbol_csv", "EURUSD_15M.csv")
    device = params.get("device", "cuda:0")
    task = params.get("task", "both")  # forecast_eval | walkforward | both
    model = params.get("model", "NeoQuasar/Kronos-small")
    tokenizer = params.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base")
    env = dict(os.environ, PYTHONPATH="src")
    if params.get("hf_token"):  # authenticate HF downloads (faster, higher rate limits)
        env["HF_TOKEN"] = params["hf_token"]
        print("HF_TOKEN set for model downloads", flush=True)

    def run(cmd: list[str], label: str) -> None:
        print(f"\nRUN [{label}]:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=work, env=env).returncode
        print(f"{label} exit code:", rc, flush=True)

    # Forecast-quality diagnostic first — cheap and decisive.
    if task in ("forecast_eval", "both"):
        # resample_list lets one job sweep timeframes (e.g. ["1h","4h","1D"]).
        # An empty entry "" means native bars.
        resamples = params.get("resample_list") or [""]
        for tf in resamples:
            tag = tf.lower() if tf else "native"
            fe_cmd = [
                sys.executable, "-m", "backtest.forecast_eval", f"data/{csv}",
                "--device", device,
                "--lookback", str(params.get("lookback", 256)),
                "--pred-len", str(params.get("pred_len", 12)),
                "--stride", str(params.get("eval_stride", 16)),
                "--sample-count", str(params.get("sample_count", 10)),
                "--threshold-bps", str(params.get("threshold_bps", 8.0)),
                "--model", model,
                "--tokenizer", tokenizer,
                "--out-json", os.path.join(WORKING, f"forecast_eval_{tag}.json"),
            ]
            if params.get("target_points") is not None:
                fe_cmd += ["--target-points", str(params["target_points"])]
            if tf:
                fe_cmd += ["--resample", tf]
            if params.get("use_volume"):
                fe_cmd.append("--use-volume")
            if params.get("cost_bps") is not None:
                fe_cmd += ["--cost-bps", str(params["cost_bps"])]
            run(fe_cmd, f"forecast_eval[{tag}]")

    if task in ("walkforward", "both"):
        cmd = [
            sys.executable,
            "-m",
            "backtest.walk_forward",
            f"data/{csv}",
            "--device",
            device,
            "--train-days",
            str(params.get("train_days", 90)),
            "--test-days",
            str(params.get("test_days", 60)),
            "--step-days",
            str(params.get("step_days", 60)),
            "--lookback",
            str(params.get("lookback", 256)),
            "--forecast-every",
            str(params.get("forecast_every", 16)),
            "--sample-count",
            str(params.get("sample_count", 10)),
            "--model",
            model,
            "--tokenizer",
            tokenizer,
            "--out-json",
            os.path.join(WORKING, "walkforward_results.json"),
        ]
        if not params.get("calibrate", False):
            cmd.append("--no-calibrate")
        run(cmd, "walk_forward")

    for html in glob.glob(os.path.join(work, "reports", "*.html")):
        shutil.copy(html, os.path.join(WORKING, os.path.basename(html)))
        print("saved", os.path.basename(html), flush=True)


if __name__ == "__main__":
    main()

# Running the walk-forward on a free Kaggle GPU

CPU does ~2 s/forecast; a Kaggle T4 GPU does it ~10–40× faster, so the full
EURUSD walk-forward drops from ~1 h to a few minutes.

There are two ways to run it. **The CLI driver is the recommended one** — it does
everything from this workspace (package → push → run → download results). The
notebook is a manual fallback.

---

## Option A — drive it from here with the Kaggle CLI (recommended)

### One-time: API token
1. Go to <https://www.kaggle.com/settings> → **Create New Token**. A
   `kaggle.json` downloads.
2. Install it:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

### Run
```bash
# packages the repo+CSVs as a Kaggle Dataset, pushes a GPU kernel, waits, downloads results
.venv/bin/python kaggle/run_on_kaggle.py --symbol EURUSD_15M.csv

# other pairs / knobs
.venv/bin/python kaggle/run_on_kaggle.py --symbol USDJPY_15M.csv --forecast-every 8 --sample-count 10

# fire-and-forget, then grab results later
.venv/bin/python kaggle/run_on_kaggle.py --push-only
.venv/bin/python kaggle/run_on_kaggle.py --fetch-only
```

Results land in `reports/kaggle/`:
- `walkforward_results.json` — per-window + aggregated OOS metrics + Monte Carlo
  (the driver also prints these to the terminal),
- `tearsheet_walkforward.html` — the QuantStats tearsheet.

What the driver does (`kaggle/run_on_kaggle.py`):
1. stages `src/ backtest/ config/ data/ pyproject.toml` + a `kaggle_params.json`
   into a Kaggle **Dataset** (`<you>/kronos-mt5-forex`),
2. writes `kaggle/kernel/kernel-metadata.json` (GPU + internet, attaches the
   dataset) and pushes the **kernel** (`kaggle/kernel/kronos_kaggle.py`),
3. polls `kaggle kernels status` until it finishes,
4. `kaggle kernels output` → `reports/kaggle/`.

---

## Option B — manual notebook

Use `kronos_walkforward.ipynb` if you'd rather click through the Kaggle UI:
1. Make a Dataset from the repo (include `src/`, `backtest/`, `config/`,
   `data/*.csv`, `pyproject.toml`; skip `.venv/` and `vendor/`).
2. New Notebook → Accelerator = **GPU T4 ×1**, Internet = **On** → Import the
   `.ipynb` → attach the dataset → set `PROJECT_INPUT` → **Run All**.
3. Download the tearsheet from the Output tab.

---

## Notes
- Internet **must** be on (pip install + Hugging Face model download).
- GPU is cheap per forecast, so defaults are finer than the CPU run:
  `forecast-every 8` (~hourly on M15) and `sample-count 10` (sharper `prob_up`).
- Free Kaggle GPU quota is ~30 h/week, ~9 h/session — ample for these runs.
- The notebook is generated from `_make_notebook.py` (regenerate with
  `python kaggle/_make_notebook.py`); don't hand-edit the `.ipynb`.

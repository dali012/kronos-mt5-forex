"""Kronos forecasting brain.

Wraps the upstream Kronos `KronosPredictor` behind a clean, engine-agnostic
interface so the strategy never imports torch or Kronos internals directly.

Upstream reference (study before implementing):
  - examples/prediction_example.py          (with volume/amount)
  - examples/prediction_wo_vol_example.py   (forex: no real volume)
  Repo: https://github.com/shiyu-coder/Kronos  (cloned by scripts/setup_kronos.sh)

Key facts to respect:
  - max_context is 512 for Kronos-small / Kronos-base (2048 for Kronos-mini).
    Keep len(df) <= max_context.
  - Input df needs columns ['open','high','low','close']; 'volume','amount'
    optional. For FX, pass tick volume as a proxy or use the volume-free path.
  - predict() is probabilistic: with sample_count > 1 you get multiple paths;
    use their distribution to estimate confidence, not just a point forecast.

Implementation note — getting the distribution:
  Upstream `KronosPredictor.predict(..., sample_count=N)` runs N sampled paths
  but *averages them internally* and only returns the mean path. That throws
  away exactly the information we want for `prob_up`. So this wrapper instead
  draws `sample_count` independent single-sample paths (each `predict(...,
  sample_count=1)` is one stochastic draw), keeps every path, and aggregates
  here: the mean path for the point forecast, and the fraction of paths closing
  above the last close for `prob_up`. Cost is `sample_count` autoregressive
  passes — fine for a once-per-bar cadence; lower `sample_count` for fast
  backtests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- Make the vendored Kronos `model` package importable -------------------
# scripts/setup_kronos.sh clones the repo to <repo>/vendor/Kronos. Add it to
# sys.path so `from model import ...` resolves without requiring the caller to
# set PYTHONPATH. A KRONOS_REPO env / constructor arg can override this.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_KRONOS_REPO = _REPO_ROOT / "vendor" / "Kronos"

_PRICE_COLS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class ForecastResult:
    """Engine-agnostic forecast output consumed by the risk/signal layer."""

    last_close: float
    horizon_close_mean: float  # mean predicted close at the horizon
    expected_return: float  # (horizon_close_mean - last_close) / last_close
    prob_up: float  # fraction of sampled paths ending above last_close
    predicted_path: pd.DataFrame  # full predicted OHLCV path, indexed by future ts
    horizon_bars: int


def _ensure_kronos_importable(kronos_repo: Path) -> None:
    repo = str(kronos_repo)
    if repo not in sys.path:
        if not (kronos_repo / "model").is_dir():
            raise FileNotFoundError(
                f"Kronos repo not found at {kronos_repo}. Run scripts/setup_kronos.sh "
                "or pass kronos_repo=... to KronosBrain."
            )
        sys.path.insert(0, repo)


class KronosBrain:
    """Loads Kronos once and produces forecasts on demand."""

    def __init__(
        self,
        tokenizer_name: str,
        model_name: str,
        device: str = "cpu",
        max_context: int = 512,
        kronos_repo: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        self.max_context = max_context
        self._tokenizer_name = tokenizer_name
        self._model_name = model_name
        self._device = device
        self._seed = seed

        repo = Path(kronos_repo) if kronos_repo is not None else _DEFAULT_KRONOS_REPO
        _ensure_kronos_importable(repo)

        # Imported lazily (after sys.path is set) so importing this module never
        # forces torch/Kronos to load until a brain is actually constructed.
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        # Kronos.predict() samples stochastically; without a seed every backtest
        # run differs. Seed once so a run is reproducible.
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        self._predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)

        # Fail loud if cuda was requested but isn't actually available, and report
        # exactly where the weights live so there's no ambiguity about CPU vs GPU.
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device={device} requested but torch.cuda.is_available() is False")
        param_device = next(self._predictor.model.parameters()).device
        mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        print(
            f"[KronosBrain] {model_name} loaded | params on {param_device} | "
            f"cuda_mem={mem:.0f}MB",
            flush=True,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        sample_count: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_volume: bool = False,
    ) -> ForecastResult:
        """Forecast `pred_len` future bars from the lookback window in `df`.

        Args:
            df: OHLCV(+amount) for the lookback window. len(df) must be <= max_context.
            x_timestamp: timestamps aligned to df rows.
            y_timestamp: timestamps for the future bars to predict (length pred_len).
            sample_count: number of independent sampled paths to draw. The point
                forecast is their mean; prob_up is the fraction closing up.
            use_volume: if False, use the volume-free prediction path (forex default).

        Returns:
            ForecastResult with expected_return and prob_up derived from the
            sampled paths.
        """
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if len(y_timestamp) != pred_len:
            raise ValueError(f"y_timestamp has {len(y_timestamp)} rows but pred_len={pred_len}")
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")

        df = df.copy()
        x_timestamp = pd.Series(pd.to_datetime(x_timestamp)).reset_index(drop=True)
        y_timestamp = pd.Series(pd.to_datetime(y_timestamp)).reset_index(drop=True)

        # No look-ahead / context cap: only the most recent max_context bars.
        if len(df) > self.max_context:
            df = df.iloc[-self.max_context :]
            x_timestamp = x_timestamp.iloc[-self.max_context :].reset_index(drop=True)

        cols = list(_PRICE_COLS)
        if use_volume:
            if "volume" not in df.columns:
                raise ValueError("use_volume=True but df has no 'volume' column")
            cols += ["volume"]
        x_df = df[cols].reset_index(drop=True)

        last_close = float(df["close"].iloc[-1])

        # Draw `sample_count` independent paths in ONE batched forward pass via
        # predict_batch (sample_count=1 per series, so each of the N identical
        # copies yields a distinct stochastic draw — same distribution as looping
        # predict(), but batched so the GPU is actually utilized). This is the key
        # speedup over a Python loop of single predictions.
        pred_dfs = self._predictor.predict_batch(
            df_list=[x_df] * sample_count,
            x_timestamp_list=[x_timestamp] * sample_count,
            y_timestamp_list=[y_timestamp] * sample_count,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=1,
            verbose=False,
        )
        close_paths = [p["close"].to_numpy(dtype=float) for p in pred_dfs]
        ohlc_paths = [p[_PRICE_COLS].to_numpy(dtype=float) for p in pred_dfs]

        close_stack = np.vstack(close_paths)  # (sample_count, pred_len)
        final_closes = close_stack[:, -1]  # close at the horizon
        mean_ohlc = np.mean(np.stack(ohlc_paths), axis=0)  # (pred_len, 4)

        horizon_close_mean = float(np.mean(final_closes))
        expected_return = (horizon_close_mean - last_close) / last_close
        prob_up = float(np.mean(final_closes > last_close))

        predicted_path = pd.DataFrame(
            mean_ohlc, columns=_PRICE_COLS, index=pd.DatetimeIndex(y_timestamp)
        )

        return ForecastResult(
            last_close=last_close,
            horizon_close_mean=horizon_close_mean,
            expected_return=expected_return,
            prob_up=prob_up,
            predicted_path=predicted_path,
            horizon_bars=pred_len,
        )

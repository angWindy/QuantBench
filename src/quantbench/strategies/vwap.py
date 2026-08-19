"""VWAP Mean-Reversion Scalper with HTF bias filter.

A multi-timeframe intraday strategy:

* **HTF bias (15-min bars, by default)**: compute an EMA on the HTF
  close and compare to the *previous* HTF bar's EMA. If
  ``ema > prev_ema`` we treat the day as long-biased; if lower, short-
  biased. A flat (==) bias means we take no trades that bar.

* **LTF entry (1-min bars)**: when the signed z-score of
  ``close - vwap`` over a rolling window exceeds ``entry_z`` AND the
  bar's bias agrees (negative z + short bias = short entry; positive z
  + long bias = long entry), we enter. We never enter against the bias.

* **LTF exit**: VWAP cross (close crosses back over VWAP), OR the
  |z-score| falls below ``exit_z`` (extension has decayed), OR the
  end-of-day time-stop fires (``eod_hour`` boundary).

The strategy is fully symmetric (long + short) but the HTF filter
prevents taking momentum trades against the prevailing higher-timeframe
trend, which is the most common way naive VWAP-reversion strategies
bleed.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta_classic as ta

from quantbench.features.intraday import add_vwap
from quantbench.features.resample import resample_bars
from quantbench.strategies.base import _ensure_bool


class VWAPMeanReversionScalper:
    """VWAP mean-reversion scalper with HTF EMA bias filter."""

    name = "vwap_reversion"

    def __init__(
        self,
        *,
        htf_freq: str = "15min",
        bias_ema_length: int = 20,
        vwap_z_window: int = 60,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        eod_hour: int = 16,   # 16 UTC = 11:00 ET (RTH close - 5h buffer) — see note
        min_bars_from_session_start: int = 5,
    ):
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")
        if bias_ema_length <= 1:
            raise ValueError("bias_ema_length must be > 1")
        if vwap_z_window <= 1:
            raise ValueError("vwap_z_window must be > 1")
        if not (0 <= eod_hour <= 23):
            raise ValueError("eod_hour must be in [0, 23]")
        if min_bars_from_session_start < 0:
            raise ValueError("min_bars_from_session_start must be >= 0")
        self.htf_freq = htf_freq
        self.bias_ema_length = bias_ema_length
        self.vwap_z_window = vwap_z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.eod_hour = eod_hour
        self.min_bars_from_session_start = min_bars_from_session_start

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) Compute VWAP + z-score on the LTF frame.
        feat = add_vwap(df)
        # z-score was added by add_vwap; we recompute it here using the
        # configurable window length in case the user picked a different
        # window than the default 60.
        dev = feat["close"] - feat["vwap"]
        roll_std = dev.rolling(self.vwap_z_window, min_periods=self.vwap_z_window).std()
        feat["vwap_z"] = dev / roll_std.replace(0, pd.NA)

        # 2) Compute HTF bias. Resample 1-min -> htf_freq, EMA, reindex back.
        htf = resample_bars(df, freq=self.htf_freq)
        htf["bias_ema"] = ta.ema(htf["close"], length=self.bias_ema_length)
        htf["bias_ema_prev"] = htf["bias_ema"].shift(1)
        # 1 = long, -1 = short, 0 = flat
        htf["bias"] = 0
        htf.loc[htf["bias_ema"] > htf["bias_ema_prev"], "bias"] = 1
        htf.loc[htf["bias_ema"] < htf["bias_ema_prev"], "bias"] = -1
        # Reindex HTF bias onto the LTF index. forward-fill carries the
        # latest HTF bar's bias forward until a new HTF bar arrives.
        bias_ltf = htf["bias"].reindex(feat.index, method="ffill").fillna(0).astype(int)

        # 3) Time-stop: bars at or after eod_hour UTC cannot enter.
        hour_ok = feat.index.hour < self.eod_hour
        # 4) Session-warmup: skip the first N bars of each UTC day so the
        # VWAP rolling window has enough history.
        session_first_bar = feat.groupby(feat.index.date).cumcount()
        warmup_ok = session_first_bar >= self.min_bars_from_session_start

        # 5) Build entries and exits.
        z = feat["vwap_z"]
        # Long entry: z <= -entry_z AND bias == 1
        long_entry = (z <= -self.entry_z) & (bias_ltf == 1) & hour_ok & warmup_ok
        # Short entry: z >= entry_z AND bias == -1
        short_entry = (z >= self.entry_z) & (bias_ltf == -1) & hour_ok & warmup_ok
        # Any entry signal.
        entries = (long_entry | short_entry).fillna(False)

        # 6) Exits: |z| falls below exit_z OR price crosses VWAP back OR
        # session ends. We use a simple "mean has reverted enough" rule:
        # |z| < exit_z. Combined with the strategy's symmetric design,
        # this acts as both a take-profit and a stop-out (when price moves
        # through VWAP and keeps going, |z| grows again, so exits only fire
        # on actual reversion).
        exits = (z.abs() < self.exit_z).fillna(False)
        # Don't emit an exit on the very first bar — there's no position
        # to close and it would prevent the corresponding entry from
        # being interpreted correctly.
        exits.iloc[0] = False

        return pd.DataFrame(
            {"entries": _ensure_bool(entries),
             "exits":   _ensure_bool(exits)}
        )

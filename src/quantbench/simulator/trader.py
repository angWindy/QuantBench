"""Paper-trading simulator.

A bar-by-bar event loop that mirrors a live broker's semantics more
closely than vectorbt's vectorized pipeline:

- The strategy sees bar *t* only after bar *t* has fully arrived.
- Signals decided at the close of bar *t* are filled at the open of
  bar *t + execution_lag_bars*, minus slippage.
- Equity is mark-to-market every bar using the bar's close.
- Fees are charged round-trip at fills.
- The simulator exposes ``step(bar) -> bool`` so it can drive a real-time
  loop (Streamlit, asyncio, Jupyter) AND a backtest replay (just feed
  the historical bars in order).

Everything the simulator does must agree with ``quantbench.backtest`` on
the same input. If they disagree, one of them is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from quantbench.backtest.engine import BacktestConfig
from quantbench.strategies.base import Strategy


@dataclass
class Position:
    """Active position state. ``size`` is in units of the instrument."""

    size: float = 0.0
    entry_price: float = 0.0
    entry_time: datetime | None = None
    entry_fee_paid: float = 0.0


@dataclass
class Fill:
    """A single trade execution."""

    time: datetime
    side: str           # "buy" | "sell"
    size: float
    price: float        # filled price (after slippage)
    fee: float
    notional: float


@dataclass
class SimState:
    """Snapshot returned by ``step(bar)`` so callers can render/stream it."""

    bar_time: datetime
    close: float
    position_size: float          # units held (>0 long, <0 short, 0 flat)
    cash: float                   # cash balance
    equity: float                 # mark-to-market equity at this bar's close
    pending_action: str | None    # "buy" | "sell" | None — order waiting in queue
    last_fill: Fill | None        # most recent fill, if any, this step


@dataclass
class SimResult:
    """Aggregated result of a full simulation run."""

    trades: pd.DataFrame
    equity_curve: pd.Series       # mark-to-market equity at each bar
    position_curve: pd.Series     # signed size at each bar
    cash_curve: pd.Series          # cash at each bar
    fills: pd.DataFrame            # every fill event
    n_bars: int
    n_fills: int
    final_equity: float


class PaperTrader:
    """Stateful paper-trading simulator.

    Usage::

        trader = PaperTrader(strategy=RandomStrategy(seed=1), config=BacktestConfig())
        state = None
        for bar in feed:
            state = trader.step(bar)
            # do something with state (log, plot, ...)

        result = trader.result()
    """

    def __init__(self, strategy: Strategy, config: BacktestConfig | None = None) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self._reset_state()

    # ----- Public API --------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state. Useful for repeated runs in a dashboard."""
        self._reset_state()

    def feed(self, df: pd.DataFrame) -> None:
        """Pre-load a DataFrame so step() can append without rebuilding.

        The strategy requires a DataFrame view, so instead of copying the
        growing list into a DataFrame on every bar (O(n^2) over a session),
        we prep an empty DataFrame and ``.loc``-append rows in O(1) amortised.
        """
        required = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        if "close" not in df.columns:
            raise ValueError("df must contain a 'close' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        cols = ["close"] + [c for c in ("open", "high", "low", "volume") if c in df.columns]
        self._df = df[cols].copy()
        self._df_seen = 0

    def step(self, bar: pd.Series) -> SimState | None:
        """Advance the simulator by one bar.

        Parameters
        ----------
        bar:
            A row from an OHLCV frame. Must have at least ``close`` (and
            ideally ``open``, ``high``, ``low``, ``volume``) and a
            DatetimeIndex in the parent frame.

        Returns
        -------
        SimState for this bar, or ``None`` if there's not enough history
        yet to compute features / decide a signal.
        """
        ts = bar.name if hasattr(bar, "name") else None
        if not isinstance(ts, pd.Timestamp):
            raise ValueError("bar must be a Series with a DatetimeIndex name")

        # Append to the pre-loaded DataFrame in O(1) instead of rebuilding
        # from a growing list of Series (which would be O(n^2) over a session).
        if self._df is not None:
            self._df_seen += 1
        else:
            self._bars.append(bar)

        # Need at least execution_lag_bars + 1 bars so the strategy can look
        # at least one bar in the past before deciding the next action.
        n_seen = self._df_seen if self._df is not None else len(self._bars)
        ready = n_seen >= max(self.config.execution_lag_bars + 1, 2)

        # 1) Execute any pending order queued at a previous bar.
        pending = self._pending
        self._pending = None
        if pending is not None and n_seen > self.config.execution_lag_bars:
            fill = self._execute(pending, ts, bar["open"])
            if fill is not None:
                self._record_fill(fill, ts)

        # 2) Let the strategy look at the *past* bars to decide a new action.
        if ready:
            if self._df is not None:
                window = self._df.iloc[: self._df_seen]
            else:
                window = pd.DataFrame(self._bars)
                window.index = pd.DatetimeIndex([b.name for b in self._bars])
            signals = self.strategy.generate_signals(window)
            # The signal at the most-recent decided bar is the one we act on.
            # We queue the action; it will fill `execution_lag_bars` bars later.
            decision_idx = -1 - self.config.execution_lag_bars
            if -decision_idx <= len(signals):
                row = signals.iloc[decision_idx]
            else:
                row = None
            if row is not None and bool(row["entries"]) and self._position.size <= 0:
                self._pending = "buy"
            elif row is not None and bool(row["exits"]) and self._position.size > 0:
                self._pending = "sell"

        # 3) Mark-to-market at this bar's close.
        equity = self._cash + self._position.size * float(bar["close"])
        self._equity_curve.append((ts, equity))
        self._position_curve.append((ts, self._position.size))
        self._cash_curve.append((ts, self._cash))

        if not ready:
            return None

        last_fill = self._fills[-1] if self._fills else None
        return SimState(
            bar_time=ts,
            close=float(bar["close"]),
            position_size=self._position.size,
            cash=self._cash,
            equity=equity,
            pending_action=self._pending,
            last_fill=last_fill,
        )

    def run(self, df: pd.DataFrame) -> SimResult:
        """Convenience: feed all bars in df sequentially and return the result."""
        self.reset()
        if "close" not in df.columns:
            raise ValueError("df must contain a 'close' column")
        self.feed(df)
        for ts, bar in df.iterrows():
            self.step(bar)
        return self.result()

    def result(self) -> SimResult:
        """Build a SimResult from the current internal state."""
        eq = pd.Series(
            {t: e for t, e in self._equity_curve}, name="equity"
        ).sort_index()
        pos = pd.Series(
            {t: p for t, p in self._position_curve}, name="position"
        ).sort_index()
        cash = pd.Series(
            {t: c for t, c in self._cash_curve}, name="cash"
        ).sort_index()
        fills_df = pd.DataFrame([vars(f) for f in self._fills]) if self._fills else pd.DataFrame()
        trades_df = self._build_trades(fills_df)

        return SimResult(
            trades=trades_df,
            equity_curve=eq,
            position_curve=pos,
            cash_curve=cash,
            fills=fills_df,
            n_bars=self._df_seen if self._df is not None else len(self._bars),
            n_fills=len(self._fills),
            final_equity=float(eq.iloc[-1]) if len(eq) else float(self.config.initial_capital),
        )

    # ----- Internals ---------------------------------------------------------

    def _reset_state(self) -> None:
        self._bars: list[pd.Series] = []
        self._df: pd.DataFrame | None = None
        self._df_seen: int = 0
        self._position = Position()
        self._cash: float = float(self.config.initial_capital)
        self._pending: str | None = None
        self._fills: list[Fill] = []
        self._equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._position_curve: list[tuple[pd.Timestamp, float]] = []
        self._cash_curve: list[tuple[pd.Timestamp, float]] = []

    def _execute(self, side: str, ts: pd.Timestamp, fill_price_ref: float) -> Fill | None:
        """Fill a pending order at bar's open +/- slippage. Returns Fill or None.

        Sizing: trade ``position_size`` fraction of equity *as measured at the
        previous close*, converted to instrument units at the fill price.
        This mirrors how a real broker would size your order at decision time.
        """
        # Use the last mark-to-market equity (i.e. cash + position * last close).
        last_eq = self._cash + self._position.size * fill_price_ref
        if last_eq <= 0:
            return None
        target_notional = last_eq * self.config.position_size

        slip = self.config.slippage_bps / 10_000
        if side == "buy":
            fill_price = fill_price_ref * (1 + slip)
            fee_rate = self.config.fee_bps / 10_000
            # Solve for size S such that S*fill_price + S*fill_price*fee_rate = target_notional
            # => S = target_notional / (fill_price * (1 + fee_rate))
            size = target_notional / (fill_price * (1 + fee_rate))
            fee = abs(size) * fill_price * fee_rate
            cost = size * fill_price + fee
            if self._cash < cost - 1e-9:
                # Not enough cash for a full target notional — skip this fill.
                return None
            self._position = Position(
                size=self._position.size + size,
                entry_price=fill_price,
                entry_time=ts,
                entry_fee_paid=fee,
            )
            self._cash -= cost
            return Fill(time=ts, side="buy", size=size, price=fill_price,
                        fee=fee, notional=size * fill_price)
        elif side == "sell":
            fill_price = fill_price_ref * (1 - slip)
            size = self._position.size  # close entire long
            proceeds = size * fill_price
            fee = abs(proceeds) * (self.config.fee_bps / 10_000)
            self._cash += proceeds - fee
            self._position = Position()
            return Fill(time=ts, side="sell", size=size, price=fill_price,
                        fee=fee, notional=proceeds)
        else:
            return None

        # Unreachable — both branches return above.
        return None  # pragma: no cover

        return Fill(time=ts, side="buy", size=size, price=fill_price,
                    fee=fee, notional=size * fill_price)

    def _record_fill(self, fill: Fill, ts: pd.Timestamp) -> None:
        self._fills.append(fill)

    def _build_trades(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        """Pair buy and sell fills into round-trip trades."""
        if fills_df.empty:
            return pd.DataFrame(
                columns=["entry_time", "exit_time", "size", "entry_price",
                         "exit_price", "fees", "pnl", "return"]
            )
        rows = []
        entry = None
        for _, f in fills_df.iterrows():
            if f["side"] == "buy" and entry is None:
                entry = f
            elif f["side"] == "sell" and entry is not None:
                rows.append({
                    "entry_time": entry["time"],
                    "exit_time": f["time"],
                    "size": float(entry["size"]),
                    "entry_price": float(entry["price"]),
                    "exit_price": float(f["price"]),
                    "fees": float(entry["fee"]) + float(f["fee"]),
                    "pnl": float(f["notional"]) - float(entry["notional"])
                           - float(entry["fee"]) - float(f["fee"]),
                    "return": 0.0,  # filled below
                })
                entry = None
        df = pd.DataFrame(rows)
        if not df.empty and "entry_price" in df.columns:
            df["return"] = df["pnl"] / (df["entry_price"] * df["size"]).replace(0, np.nan)
        return df

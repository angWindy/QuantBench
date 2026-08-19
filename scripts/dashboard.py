"""Streamlit dashboard for QuantBench paper-trading sessions.

Launches a multi-page UI that:

- Lists every session JSON under ``reports/sessions/``.
- For a selected session, plots the equity curve, drawdown, and a
  metrics table (Sharpe, Sortino, MaxDD, Calmar, Win Rate, ...).
- Lets you overlay multiple sessions on the same equity chart for
  side-by-side comparison.
- Shows the trade-level table with click-to-filter.

Usage:
    streamlit run scripts/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = REPO_ROOT / "reports" / "sessions"
DARK_BG = "#0E1117"
ACCENT = "#00CC96"
RISK = "#EF553B"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_sessions() -> list[Path]:
    """Return all session JSON files under reports/sessions/, newest first."""
    if not SESSIONS_DIR.exists():
        return []
    return sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_session(path: str) -> dict:
    """Load a session JSON file. Returns a dict with sidecar CSV paths."""
    p = Path(path)
    payload = json.loads(p.read_text())
    payload["_path"] = str(p)
    payload["_equity_csv"] = str(p.with_suffix(".equity.csv"))
    payload["_trades_csv"] = str(p.with_suffix(".trades.csv"))
    return payload


@st.cache_data(show_spinner=False)
def load_equity(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df["equity"]
    s.index = pd.to_datetime(s.index, utc=True)
    s.name = "equity"
    return s


@st.cache_data(show_spinner=False)
def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _figure_layout(title: str) -> dict:
    return dict(
        title=title,
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="white"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=400,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )


def render_equity_chart(equity: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                             mode="lines", name="Equity", line=dict(color=ACCENT, width=2)))
    fig.update_layout(**_figure_layout(title))
    fig.update_yaxes(title_text="Equity ($)")
    fig.update_xaxes(title_text="Time (UTC)")
    return fig


def render_drawdown_chart(equity: pd.Series, title: str) -> go.Figure:
    dd = _drawdown(equity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100,
                             mode="lines", name="Drawdown",
                             fill="tozeroy",
                             line=dict(color=RISK, width=1.5)))
    fig.update_layout(**_figure_layout(title))
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Time (UTC)")
    return fig


def render_comparison(equities: dict[str, pd.Series], title: str) -> go.Figure:
    """Normalise every series to 100 at start and overlay on the same scale."""
    fig = go.Figure()
    for name, eq in equities.items():
        norm = (eq / eq.iloc[0] * 100).copy()
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values,
                                 mode="lines", name=name, line=dict(width=2)))
    fig.update_layout(**_figure_layout(title))
    fig.update_yaxes(title_text="Equity (start=100)")
    fig.update_xaxes(title_text="Time (UTC)")
    return fig


def render_metrics_table(metrics: dict) -> pd.DataFrame:
    """Format the metrics dict as a nice 2-column DataFrame."""
    pretty = {
        "total_return": "Total Return",
        "cagr": "CAGR",
        "sharpe": "Sharpe Ratio",
        "sortino": "Sortino Ratio",
        "max_drawdown": "Max Drawdown",
        "calmar": "Calmar Ratio",
        "win_rate": "Win Rate",
        "profit_factor": "Profit Factor",
        "n_trades": "Trades",
    }
    pct = {"total_return", "cagr", "max_drawdown", "win_rate"}
    rows = []
    for key, label in pretty.items():
        if key not in metrics:
            continue
        v = metrics[key]
        if key in pct:
            display = f"{v:.2%}"
        elif key == "n_trades":
            display = f"{int(v):,}"
        else:
            display = f"{v:.3f}"
        rows.append({"Metric": label, "Value": display})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def section_overview(session: dict) -> None:
    cfg = session["config"]
    st.subheader(f"Session: {session['name']}")
    cols = st.columns(4)
    cols[0].metric("Final Equity", f"${session['final_equity']:,.2f}")
    cols[1].metric("Total Return", f"{session['metrics']['total_return']:.2%}")
    cols[2].metric("Bars", f"{session['n_bars']:,}")
    cols[3].metric("Fills", f"{session['n_fills']:,}")
    st.caption(
        f"Strategy: `{cfg['strategy']}({cfg['strategy_params']})` · "
        f"fees {cfg['fee_bps']} bps · slippage {cfg['slippage_bps']} bps · "
        f"lag {cfg['execution_lag_bars']} bar · "
        f"capital ${cfg['initial_capital']:,.0f} · "
        f"window {session['data_summary']['start']} → {session['data_summary']['end']}"
    )


def section_curves(session: dict) -> None:
    eq = load_equity(session["_equity_csv"])
    left, right = st.columns(2)
    with left:
        st.plotly_chart(render_equity_chart(eq, "Equity curve"), width="stretch")
    with right:
        st.plotly_chart(render_drawdown_chart(eq, "Drawdown"), width="stretch")


def section_metrics(session: dict) -> None:
    st.subheader("Performance metrics")
    st.table(render_metrics_table(session["metrics"]))


def section_trades(session: dict) -> None:
    st.subheader("Trades")
    try:
        trades = load_trades(session["_trades_csv"])
    except FileNotFoundError:
        st.info("No trades for this session.")
        return
    if trades.empty:
        st.info("No trades for this session.")
        return
    resample = {
        "entry_time": "min",
        "exit_time": "max",
        "size": "sum",
        "fees": "max",
        "pnl": "sum",
    }
    sort_choice = st.selectbox("Sort trades by",
                               ["exit_time", "pnl", "return"],
                               index=0)
    ascending = st.checkbox("Ascending", value=False)
    view = trades.sort_values(sort_choice, ascending=ascending)
    st.dataframe(view, width="stretch", hide_index=True)
    win = (trades["pnl"] > 0).sum()
    st.caption(f"Wins: {win} · Losses: {len(trades) - win} · "
               f"Total PnL: ${trades['pnl'].sum():,.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="QuantBench Dashboard",
                       page_icon="chart_with_upwards_trend",
                       layout="wide")
    st.title("QuantBench Paper-Trading Dashboard")

    sessions = list_sessions()
    if not sessions:
        st.warning(
            "No sessions found under `reports/sessions/`. "
            "Run `python scripts/run_paper_session.py` first."
        )
        return

    # Sidebar: single-session selector + multi-session comparison selector.
    with st.sidebar:
        st.header("Sessions")
        all_names = [p.stem for p in sessions]
        selected_idx = st.selectbox(
            "Active session",
            range(len(sessions)),
            format_func=lambda i: all_names[i],
        )
        st.markdown("---")
        st.header("Compare")
        compare_names = st.multiselect(
            "Overlay these sessions",
            options=all_names,
            default=[],
        )

    session = load_session(str(sessions[selected_idx]))
    section_overview(session)
    section_curves(session)

    # Side breakdown (buy vs sell fills) — useful for long/short-aware strategies.
    breakdown = session.get("side_breakdown")
    if breakdown:
        bc1, bc2 = st.columns(2)
        bc1.metric("Buy fills (open long)", breakdown.get("buy", 0))
        bc2.metric("Sell fills (close long)", breakdown.get("sell", 0))

    if compare_names:
        st.markdown("---")
        st.subheader("Comparison")
        # Map selected names back to paths.
        name_to_path = {p.stem: str(p) for p in sessions}
        equities = {}
        for name in compare_names:
            other = load_session(name_to_path[name])
            try:
                equities[name] = load_equity(other["_equity_csv"])
            except FileNotFoundError:
                continue
        if equities:
            st.plotly_chart(
                render_comparison(equities, "Equity curves (normalised to 100)"),
                width="stretch",
            )

    st.markdown("---")
    section_metrics(session)
    section_trades(session)

    st.markdown("---")
    st.caption(
        "QuantBench · data: `data/processed/NQ_1min.parquet` · "
        "session files governed by `reports/sessions/.gitignore`"
    )


if __name__ == "__main__":
    main()

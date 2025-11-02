# file: C3_Visualization.py
# purpose: candlestick and boxplot visualizations for stock OHLCV data

from typing import Optional, Iterable, Literal, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# ---------------------------------------------------------------------------------
# helper: fold consecutive trading days into n-day candlesticks
# ---------------------------------------------------------------------------------
def aggregate_into_n_trading_days(
    df: pd.DataFrame,
    n: int = 1,
    ohlc_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
    volume_col: str = "volume",
    keep_other_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """ aggregates consecutive trading days into blocks of size n using financial OHLC rules.

    idea:
      instead of looping, build vectorized “blocks” of size n and apply first/max/min/last rules.
      this keeps things fast and consistent even for long datasets.

    parameters:
      df: dataframe indexed by datetimeindex in ascending order
      n: number of rows per block (like 5 for weekly candles)
      keep_other_cols: optional extra columns, each carrying its last non-null value

    returns:
      a new dataframe, each row representing one n-day candle labeled by its last trading date """
    if n < 1:
        raise ValueError("n must be >= 1")

    # lowercase headers once to avoid mismatch between Open/open
    if not set(ohlc_cols).issubset(df.columns.str.lower()):
        df = df.rename(columns={c: c.lower() for c in df.columns})

    o, h, l, c = ohlc_cols
    work = df.copy()

    # time index sanity check
    if not isinstance(work.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")
    work = work.sort_index()

    # vectorized block ids (0..0,1..1,2..2 per group)
    work["_block"] = np.arange(len(work)) // n

    # financial aggregation rules
    agg_map = {o: "first", h: "max", l: "min", c: "last"}
    if volume_col in work.columns:
        agg_map[volume_col] = "sum"

    # handle any extra columns by carrying the last non-null value
    if keep_other_cols:
        for extra in keep_other_cols:
            if extra in work.columns and extra not in agg_map:
                agg_map[extra] = lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan

    grouped = work.groupby("_block")

    # index each block by its last trading date
    last_dates = grouped.apply(lambda g: g.index.max())

    # mixed aggregation (different rules per column)
    agg = grouped.agg(agg_map)
    agg.index = pd.DatetimeIndex(last_dates.values, name=work.index.name)

    return agg.sort_index()

# ---------------------------------------------------------------------------------
# candlestick plotter with optional n-day aggregation
# ---------------------------------------------------------------------------------
def plot_candlesticks(
    df: pd.DataFrame,
    n: int = 1,
    style: str = "yahoo",
    title: Optional[str] = None,
    mav: Optional[Iterable[int]] = None,
    volume: bool = True,
    tight_layout: bool = True,
    returnfig: bool = False,
    adjust: Optional[dict] = None,
):
    """ draws OHLCV candlesticks, optionally after grouping rows into n-day blocks.

    what it actually does:
      - normalizes headers to lowercase
      - enforces datetime index
      - applies n-day aggregation if requested
      - coerces OHLCV columns to numeric
      - optionally returns the matplotlib figure instead of showing it """
    work = df.copy().rename(columns={c: c.lower() for c in df.columns})

    # enforce proper datetime index
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, errors="coerce")
    work = work.loc[work.index.notna()].sort_index()

    # aggregate if needed
    if n > 1:
        work = aggregate_into_n_trading_days(work, n=n)

    # coerce OHLCV columns to numeric, drop broken rows
    for col in ("open", "high", "low", "close", "volume"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    needed = [c for c in ("open", "high", "low", "close") if c in work.columns]
    work = work.dropna(subset=needed)

    if title is None:
        title = f"candlesticks - {n}-day" if n > 1 else "candlesticks - 1-day"

    kwargs = dict(
        type="candle",
        volume=volume,
        style=style,
        title=title,
        tight_layout=tight_layout,
        show_nontrading=False,
    )
    if mav is not None:
        kwargs["mav"] = mav

    if returnfig:
        # return figure so caller can manage multiple windows and one plt.show()
        fig, axlist = mpf.plot(work, returnfig=True, **kwargs)
        if adjust:
            try:
                fig.subplots_adjust(**adjust)
            except Exception:
                pass
        return fig, axlist
    else:
        mpf.plot(work, **kwargs)
        return None

# ---------------------------------------------------------------------------------
# boxplot over non-overlapping n-day windows
# ---------------------------------------------------------------------------------
WindowMetric = Literal["close", "returns", "range", "hl_pct"]

def _window_slices(values: np.ndarray, n: int) -> List[np.ndarray]:
    """ splits a 1D array into full windows of length n

    last partial window is dropped on purpose
    (inconsistent block sizes would skew boxplots) """
    full_blocks = len(values) // n
    return [values[i * n : (i + 1) * n] for i in range(full_blocks)]

def plot_boxplot_moving_window(
    df: pd.DataFrame,
    n: int = 5,
    metric: WindowMetric = "returns",
    title: Optional[str] = None,
    label_every: int = 3,
    make_new_figure: bool = False,
) -> None:
    """ draws a boxplot where each box shows distribution stats over one n-day window

    metrics explained:
      close     raw close prices per day
      returns   fractional day-to-day change (close to close)
      range     intraday difference (high-low)
      hl_pct    intraday range divided by close

    small details:
      - types are coerced to numeric at start to avoid crashes
      - returns metric drops first NaN from pct_change
      - box labels use the last date in each window for readability """
    work = df.copy().rename(columns={c: c.lower() for c in df.columns})

    # ensure datetime index and sorted order
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, errors="coerce")
    work = work.loc[work.index.notna()].sort_index()

    # coerce needed columns to numeric
    for col in ("open", "high", "low", "close"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # pick per-day data depending on metric
    if metric == "close":
        per_day = work["close"].astype(float).to_numpy()
        idx = work.index
    elif metric == "returns":
        close_num = work["close"].astype(float)
        per_day = close_num.pct_change().dropna().to_numpy()
        idx = close_num.index[1:]
    elif metric == "range":
        per_day = (work["high"].astype(float) - work["low"].astype(float)).to_numpy()
        idx = work.index
    elif metric == "hl_pct":
        close_num = work["close"].astype(float)
        per_day = ((work["high"].astype(float) - work["low"].astype(float)) / close_num).to_numpy()
        idx = work.index
    else:
        raise ValueError("unsupported metric")

    slices = _window_slices(per_day, n)
    if len(slices) == 0:
        raise ValueError("not enough rows to form even one full window")

    # each box label = last date in its block
    block_dates = [idx[min((i + 1) * n, len(idx)) - 1] for i in range(len(slices))]
    labels = [d.strftime("%Y-%m-%d") for d in block_dates]

    if make_new_figure:
        plt.figure(figsize=(10, 5))

    # draw the boxplot with simple readable styling
    plt.boxplot(
        slices,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="#cfe8ff", edgecolor="#1f77b4", linewidth=1.2),
        medianprops=dict(color="#d62728", linewidth=1.5),
        whiskerprops=dict(color="#1f77b4", linewidth=1.0),
        capprops=dict(color="#1f77b4", linewidth=1.0),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#ff7f0e",
                        markeredgecolor="#ff7f0e", alpha=0.6),
    )

    plt.title(title or f"boxplot - {metric} over {n}-day blocks")
    plt.xlabel(f"consecutive {n}-day blocks")
    plt.ylabel({
        "close": "close price",
        "returns": "daily return (fraction)",
        "range": "high - low",
        "hl_pct": "(high - low) / close",
    }[metric])

    xticks = np.arange(1, len(labels) + 1)
    xtick_labels = [labels[i - 1] if (i - 1) % label_every == 0 else "" for i in range(len(xticks))]
    plt.xticks(xticks, xtick_labels, rotation=45, ha="right")
    plt.tight_layout()

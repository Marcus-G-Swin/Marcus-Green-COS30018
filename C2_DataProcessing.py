# File: C2_DataProcessing.py
# purpose: task 2 data loading and preprocessing used by the stock prediction scripts

from __future__ import annotations

import os
import pickle
import random
from collections import deque
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn import preprocessing
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from yahoo_fin import stock_info as si

# set stable seeds so runs are reproducible
_DEFAULT_SEED = 314
np.random.seed(_DEFAULT_SEED)
tf.random.set_seed(_DEFAULT_SEED)
random.seed(_DEFAULT_SEED)

__all__ = [
    "load_and_process_data",
    "load_data",
    "create_model",
    "shuffle_in_unison",
]


# small helper to keep two arrays shuffled in the exact same way
def shuffle_in_unison(a: np.ndarray, b: np.ndarray, seed: Optional[int] = None) -> None:
    """ applies the same random permutation to a and b so rows remain aligned
    handy for shuffling train pairs without breaking x to y mapping """
    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(a))
    else:
        idx = np.random.permutation(len(a))
    a[:] = a[idx]
    b[:] = b[idx]


# build a cache path that uniquely keys by ticker and date window
def _cache_path(cache_dir: Optional[str], ticker: str,
                start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    sd = (start_date or "start").replace("-", "")
    ed = (end_date or "end").replace("-", "")
    return os.path.join(cache_dir, f"{ticker}_{sd}_{ed}.csv")


# force a datetime index and make sure it is sorted
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    lots of sources ship either a datetime index or a date column
    accept both, coerce to DatetimeIndex, then sort ascending
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    # common date column names from different sources
    for name in ("date", "Date", "DATE", "timestamp", "Timestamp"):
        if name in df.columns:
            dt = pd.to_datetime(df[name], errors="coerce")
            mask = dt.notna()
            out = df.loc[mask].copy()
            out.index = dt[mask]
            return out.sort_index()

    # as a last resort try to parse the existing index
    dt = pd.to_datetime(df.index, errors="coerce")
    mask = dt.notna()
    out = df.loc[mask].copy()
    out.index = dt[mask]
    return out.sort_index()


# pick a scaler based on a short string, so cli flags stay simple
def _mk_scaler(kind: str):
    if kind == "minmax":
        return preprocessing.MinMaxScaler()
    if kind == "standard":
        return preprocessing.StandardScaler()
    if kind == "robust":
        return preprocessing.RobustScaler()
    raise ValueError("scaler_type must be minmax, standard, or robust")


def load_data(
    ticker: Union[str, pd.DataFrame],
    n_steps: int = 50,
    scale: bool = True,
    shuffle: bool = True,
    lookup_step: int = 1,
    split_by_date: bool = True,
    test_size: float = 0.2,
    feature_columns: List[str] = None,
    # task 2 additions
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    nan_strategy: str = "ffill",
    nan_fill_value: float = 0.0,
    cache_dir: Optional[str] = "./cache",
    prefer_cache: bool = True,
    force_refresh: bool = False,
    scaler_type: str = "minmax",
    save_scalers_path: Optional[str] = None,
    seed: int = _DEFAULT_SEED,
    # compatibility with older callers
    split_method: Optional[str] = None,
    split_date: Optional[str] = None,
) -> Dict[str, object]:
    """ loads prices, cleans them, scales features, builds rolling windows, and splits into train and test

    returns a dict with:
      df                 raw unscaled frame
      X_train, y_train   training arrays with shapes (samples, n_steps, n_features) and (samples, 1)
      X_test,  y_test    test arrays with matching shapes
      test_df            unscaled rows aligned one to one with X_test rows
      last_sequence      window that ends at the dataset tail for a next step prediction
      column_scaler      dict of per column scalers if scale=True
      feature_columns    the exact feature order used to build X arrays """
    if feature_columns is None:
        feature_columns = ['adjclose', 'volume', 'open', 'high', 'low']

    # local seeds so a forgetful caller still gets deterministic behavior
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # 1) fetch data from cache or network
    if isinstance(ticker, str):
        csv_path = _cache_path(cache_dir, ticker, start_date, end_date)
        if csv_path and prefer_cache and os.path.exists(csv_path) and not force_refresh:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            # try yahoo_fin first, then fall back to yfinance
            try:
                df = si.get_data(ticker)
            except Exception:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                )
                if df.empty:
                    raise RuntimeError(f"could not fetch data for {ticker}. check the symbol and network.")
                # standardize headers to lower case so downstream lookups are simple
                df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjclose",
                    "Volume": "volume",
                }, inplace=True)
                # some feeds do not include adjclose, fall back to close
                if "adjclose" not in df.columns:
                    if "close" in df.columns:
                        df["adjclose"] = df["close"]
                    else:
                        raise ValueError("no 'adjclose' or 'close' column present in data.")

            # date windowing happens after fetch in case the upstream source ignores dates
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if csv_path:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker.copy()
    else:
        raise TypeError("ticker must be a str or a pandas DataFrame")

    df = _ensure_datetime_index(df)

    result: Dict[str, object] = {"df": df.copy()}

    # 2) feature checks and carry a date column for alignment later
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if "date" not in df.columns:
        df["date"] = df.index

    # coerce to numeric and guard against accidental multi column objects
    for col in feature_columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 1:
                s = s.iloc[:, 0]
            else:
                raise ValueError(f"feature '{col}' has {s.shape[1]} subcolumns")
        df[col] = pd.to_numeric(s, errors="coerce")

    # 3) fill or drop missing values per the chosen strategy
    if nan_strategy == "drop":
        df = df.dropna()
    elif nan_strategy == "ffill":
        df = df.ffill().dropna()
    elif nan_strategy == "bfill":
        df = df.bfill().dropna()
    elif nan_strategy == "interpolate":
        df = df.interpolate(method="time").dropna()
    elif nan_strategy == "constant":
        df = df.fillna(nan_fill_value)
    else:
        raise ValueError("nan_strategy must be one of: drop, ffill, bfill, interpolate, constant")

    # 4) fit a scaler per feature, not one scaler for all
    # this keeps inverse_transform simple when we only want to invert adjclose
    if scale:
        column_scaler: Dict[str, preprocessing.MinMaxScaler] = {}
        for column in feature_columns:
            vals = df[column].to_numpy().reshape(-1, 1)
            scaler = _mk_scaler(scaler_type)
            df[column] = scaler.fit_transform(vals)
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler
        if save_scalers_path:
            os.makedirs(os.path.dirname(save_scalers_path), exist_ok=True)
            with open(save_scalers_path, "wb") as f:
                pickle.dump(column_scaler, f)

    # 5) create the next step target and remember the very tail for convenience
    df["future"] = df["adjclose"].shift(-lookup_step)
    last_tail = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)

    # 6) build rolling windows of length n_steps
    sequence_data: List[List[Union[np.ndarray, float]]] = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df["future"].values):
        sequences.append(entry)                   # append current row into the moving window
        if len(sequences) == n_steps:
            window = np.array(sequences)         # shape: (n_steps, n_features + 1 with date)
            sequence_data.append([window, target])

    # last_sequence is the tail window used by some callers to forecast one extra step
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_tail)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result["last_sequence"] = last_sequence

    # unpack into X and y and also grab the end date for each window
    X = np.array([seq for seq, _ in sequence_data])
    y = np.array([t for _, t in sequence_data]).reshape(-1, 1)
    end_dates = X[:, -1, -1]                      # the date column we carried inside the window

    # 7) split into train and test based on the chosen mode
    mode = (split_method or ("date" if split_by_date else "random")).lower()
    if mode not in {"date", "ratio", "random"}:
        raise ValueError("split_method must be 'date', 'ratio', or 'random'")

    if mode == "ratio":
        n_train = int((1.0 - test_size) * len(X))
        X_train, y_train = X[:n_train], y[:n_train]
        X_test,  y_test  = X[n_train:], y[n_train:]
        test_dates = end_dates[n_train:]
        if shuffle:
            shuffle_in_unison(X_train, y_train, seed=seed)

    elif mode == "date":
        if not split_date:
            raise ValueError("split_date must be provided when split_method == 'date'")
        split_ts = pd.to_datetime(split_date)
        train_mask = end_dates < np.datetime64(split_ts)
        test_mask = ~train_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        test_dates = end_dates[test_mask]
        if shuffle:
            shuffle_in_unison(X_train, y_train, seed=seed)

    else:
        idx = np.arange(len(X))
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        cut = int((1.0 - test_size) * len(idx))
        train_idx, test_idx = idx[:cut], idx[cut:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        test_dates = end_dates[test_idx]
        if shuffle:
            shuffle_in_unison(X_train, y_train, seed=seed)

    # 8) build a test dataframe that lines up exactly with rows in X_test
    test_df = result["df"].reindex(test_dates)
    test_df = test_df[~test_df.index.duplicated(keep="first")]
    result["test_df"] = test_df

    # 9) drop the carried date column from X and cast to float32 for keras
    feat_count = len(feature_columns)
    X_train = X_train[:, :, :feat_count].astype(np.float32)
    X_test  = X_test[:,  :, :feat_count].astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    result.update({
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_columns": list(feature_columns),
    })
    return result


def load_and_process_data(**kwargs):
    """ backward compatible alias for older imports """
    return load_data(**kwargs)


# kept here for api compatibility with older notebooks that import create_model
# it mirrors the v01 behavior and is intentionally simple
def create_model(sequence_length: int, n_features: int) -> Sequential:
    """ minimal lstm stack that matches the original helper
    this is provided for older callers that expect create_model in this module """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer="rmsprop", metrics=["mae"])
    return model

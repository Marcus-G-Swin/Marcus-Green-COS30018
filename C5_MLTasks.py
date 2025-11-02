import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

from C2_DataProcessing import load_and_process_data
from C4_ModelMaker import create_dl_model, parse_layer_sizes
from C3_Visualization import plot_candlesticks, plot_boxplot_moving_window


# helper: invert scaling for each horizon column separately
def inverse_each_horizon(mat_scaled: np.ndarray, scaler) -> np.ndarray:
    """ undoes scaling for multistep predictions, one horizon at a time.
    sklearn scalers expect a single column, so we loop column by column.
    otherwise, inverse_transform would squash all horizons together. """
    cols = []
    for i in range(mat_scaled.shape[1]):        # go horizon by horizon
        col = scaler.inverse_transform(mat_scaled[:, i:i+1])  # inverse one column
        cols.append(col.ravel())                # flatten back to 1d
    return np.column_stack(cols)                # rejoin into full matrix


def make_multistep_labels(y: np.ndarray, k: int) -> np.ndarray:
    """ turns a single-step target into k-step ahead targets.
    each row now contains the next k true values.
    this is a classic direct multistep setup for time series forecasting. """
    sequences = []
    for i in range(len(y) - k + 1):             # slide a window across the target series
        sequences.append(y[i : i + k].ravel())  # grab k consecutive future points
    return np.array(sequences, dtype=np.float32)


# ------------------------------
# Core trainers used internally
# ------------------------------
def train_multistep_model(
    X_train, y_train, X_test, y_test, test_df, scalers,
    sequence_length: int,
    n_features: int,
    k: int = 5,
    cell: str = "LSTM",
    layer_sizes=(128, 64),
    dropout=0.3,
    bidirectional=False,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    epochs: int = 25,
    batch_size: int = 32,
):
    """ trains a recurrent model that predicts the next k closing prices in one shot. """
    # reshape y to contain k future steps per sample
    y_train_k = make_multistep_labels(y_train, k)
    y_test_k  = make_multistep_labels(y_test, k)

    # match input length to shorter target array
    X_train_k = X_train[: len(y_train_k)]
    X_test_k  = X_test[: len(y_test_k)]

    # reuse task 4 model builder
    model = create_dl_model(
        sequence_length=sequence_length,
        n_features=n_features,
        layer_type=cell,
        layer_sizes=layer_sizes,
        dropout=dropout,
        bidirectional=bidirectional,
        loss=loss,
        optimizer=optimizer,
        metrics=("mae",),
    )

    # swap the Dense(1) output for Dense(k)
    model.pop()
    model.add(Dense(k, activation="linear"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

    # standard training loop
    hist = model.fit(X_train_k, y_train_k, epochs=epochs, batch_size=batch_size, verbose=1)
    preds = model.predict(X_test_k, verbose=0)

    # bring predictions and actuals back to original price scale
    adj_scaler = scalers.get("adjclose")
    preds_inv  = inverse_each_horizon(preds, adj_scaler)
    actual_inv = inverse_each_horizon(y_test_k, adj_scaler)

    # compute error in actual price units
    mae_val = mean_absolute_error(actual_inv, preds_inv)
    print(f"[Task5-Multistep] Test MAE (price scale): {mae_val:.4f}")

    # quick multi-horizon plot
    plt.figure(figsize=(10,5))
    for h in range(k):
        style = "--" if h > 0 else "-"
        plt.plot(preds_inv[:, h], label=f"Predicted (t+{h+1})", linestyle=style)
    plt.plot(actual_inv[:, 0], label="Actual (t+1)", linewidth=2)
    plt.title(f"Multistep Prediction (k={k})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    # qualitative visuals
    ohlcv_test = pd.DataFrame({
        "open": test_df["open"].values.ravel(),
        "high": test_df["high"].values.ravel(),
        "low":  test_df["low"].values.ravel(),
        "close": test_df["close"].values.ravel(),
        "volume": test_df["volume"].values.ravel(),
    }, index=pd.DatetimeIndex(test_df.index).sort_values())

    plot_candlesticks(ohlcv_test, n=5, title="Task5 multistep candlesticks", volume=False)
    plot_boxplot_moving_window(ohlcv_test, n=5, metric="returns", title="Task5 multistep boxplot")
    plt.show()

    return model, hist, preds_inv, actual_inv


def train_multivariate_model(
    bundle: dict,
    cell="LSTM",
    layer_sizes=(128, 64),
    dropout=0.3,
    bidirectional=False,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    epochs=25,
    batch_size=32,
):
    """ trains a model that uses multiple input features to predict next-day close. """
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_test  = bundle["X_test"]
    y_test  = bundle["y_test"]
    test_df = bundle["test_df"]
    scalers = bundle["column_scaler"]

    model = create_dl_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        layer_type=cell,
        layer_sizes=layer_sizes,
        dropout=dropout,
        bidirectional=bidirectional,
        loss=loss,
        optimizer=optimizer,
        metrics=("mae",),
    )

    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    preds = model.predict(X_test, verbose=0)

    adj_scaler = scalers.get("adjclose")
    preds_inv = adj_scaler.inverse_transform(preds)
    actual_inv = test_df["adjclose"].values.reshape(-1,1)

    mae_val = mean_absolute_error(actual_inv, preds_inv)
    print(f"[Task5-Multivariate] Test MAE (price scale): {mae_val:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(actual_inv, label="Actual")
    plt.plot(preds_inv, label="Predicted")
    plt.title("Multivariate Prediction (all features)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ohlcv_test = pd.DataFrame({
        "open": test_df["open"].values.ravel(),
        "high": test_df["high"].values.ravel(),
        "low":  test_df["low"].values.ravel(),
        "close": test_df["close"].values.ravel(),
        "volume": test_df["volume"].values.ravel(),
    }, index=pd.DatetimeIndex(test_df.index).sort_values())

    plot_candlesticks(ohlcv_test, n=5, title="Task5 multivariate candlesticks", volume=False)
    plot_boxplot_moving_window(ohlcv_test, n=5, metric="returns", title="Task5 multivariate boxplot")
    plt.show()

    return model, hist, preds_inv, actual_inv


def train_multivariate_multistep_model(
    bundle: dict,
    k: int = 5,
    cell="LSTM",
    layer_sizes=(128, 64),
    dropout=0.3,
    bidirectional=False,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    epochs=25,
    batch_size=32,
):
    """ multivariate plus multistep trainer. """
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_test  = bundle["X_test"]
    y_test  = bundle["y_test"]
    test_df = bundle["test_df"]
    scalers = bundle["column_scaler"]

    y_train_k = make_multistep_labels(y_train, k)
    y_test_k  = make_multistep_labels(y_test, k)
    X_train_k = X_train[: len(y_train_k)]
    X_test_k  = X_test[: len(y_test_k)]

    model = create_dl_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        layer_type=cell,
        layer_sizes=layer_sizes,
        dropout=dropout,
        bidirectional=bidirectional,
        loss=loss,
        optimizer=optimizer,
        metrics=("mae",),
    )
    model.pop()
    model.add(Dense(k, activation="linear"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

    hist = model.fit(X_train_k, y_train_k, epochs=epochs, batch_size=batch_size, verbose=1)
    preds = model.predict(X_test_k, verbose=0)

    adj_scaler = scalers.get("adjclose")
    preds_inv  = inverse_each_horizon(preds, adj_scaler)
    actual_inv = inverse_each_horizon(y_test_k, adj_scaler)

    mae_val = mean_absolute_error(actual_inv, preds_inv)
    print(f"[Task5-Multivariate-Multistep] Test MAE (price scale): {mae_val:.4f}")

    plt.figure(figsize=(10,5))
    for h in range(k):
        style = "--" if h > 0 else "-"
        plt.plot(preds_inv[:, h], label=f"Predicted (t+{h+1})", linestyle=style)
    plt.plot(actual_inv[:, 0], label="Actual (t+1)", linewidth=2)
    plt.title(f"Multivariate Multistep Prediction (k={k})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    ohlcv_test = pd.DataFrame({
        "open": test_df["open"].values.ravel(),
        "high": test_df["high"].values.ravel(),
        "low":  test_df["low"].values.ravel(),
        "close": test_df["close"].values.ravel(),
        "volume": test_df["volume"].values.ravel(),
    }, index=pd.DatetimeIndex(test_df.index).sort_values())

    plot_candlesticks(ohlcv_test, n=5, title="Task5 multi-var+step candlesticks", volume=False)
    plot_boxplot_moving_window(ohlcv_test, n=5, metric="returns", title="Task5 multi-var+step boxplot")
    plt.show()

    return model, hist, preds_inv, actual_inv


# --------------------------------------------------------
# Thin CLI-style wrappers so v05 imports do not break
# These accept the argparse args namespace from v05
# --------------------------------------------------------
def _load_bundle_from_args(args, feature_columns):
    return load_and_process_data(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        n_steps=args.n_steps,
        lookup_step=args.lookup_step,
        feature_columns=feature_columns,
        nan_strategy="ffill",
        split_method="date",
        split_date=args.test_start,
        shuffle=False,
        cache_dir=args.cache_dir if hasattr(args, "cache_dir") else "./cache",
        prefer_cache=True,
        force_refresh=getattr(args, "force_refresh", False),
        scale=True,
        scaler_type=getattr(args, "scaler_type", "minmax"),
        save_scalers_path=f"{getattr(args, 'cache_dir', './cache')}/{args.ticker}_scalers.pkl",
        seed=getattr(args, "seed", 42),
    )

def train_one_step_baseline(args):
    """Keeps v05 import stable - simple 1 step trainer compatible with args."""
    features = ["adjclose"]
    bundle = _load_bundle_from_args(args, features)
    X_train = bundle["X_train"]; y_train = bundle["y_train"]
    X_test = bundle["X_test"];   y_test = bundle["y_test"]
    scaler = bundle["column_scaler"]["adjclose"]

    model = create_dl_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        layer_type=args.cell,
        layer_sizes=parse_layer_sizes(args.layers) or [128, 64],
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        loss=args.loss,
        optimizer=args.optimizer,
        metrics=("mae",),
    )
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    preds = model.predict(X_test, verbose=0)
    y_true = scaler.inverse_transform(preds*0 + y_test.reshape(-1,1)).ravel()
    y_pred = scaler.inverse_transform(preds).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(f"[Task5-OneStep] Test MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    return model, preds

def train_multistep_k(args, k: int = 5):
    """Wrapper that builds data from args and calls the multistep trainer."""
    features = ["adjclose"]
    bundle = _load_bundle_from_args(args, features)
    model, hist, preds_inv, actual_inv = train_multistep_model(
        X_train=bundle["X_train"],
        y_train=bundle["y_train"],
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],
        test_df=bundle["test_df"],
        scalers=bundle["column_scaler"],
        sequence_length=bundle["X_train"].shape[1],
        n_features=bundle["X_train"].shape[2],
        k=k,
        cell=args.cell,
        layer_sizes=parse_layer_sizes(args.layers) or (128, 64),
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        loss=args.loss,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    return model, hist, preds_inv, actual_inv

def train_multivariate_one_step(args):
    """Wrapper that builds a multivariate bundle and calls the multivariate trainer."""
    features = ["adjclose", "open", "high", "low", "volume"]
    bundle = _load_bundle_from_args(args, features)
    return train_multivariate_model(
        bundle=bundle,
        cell=args.cell,
        layer_sizes=parse_layer_sizes(args.layers) or (128, 64),
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        loss=args.loss,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

def train_multivariate_multistep(args, k: int = 5):
    """Wrapper that builds a multivariate bundle and calls the multi-multi trainer."""
    features = ["adjclose", "open", "high", "low", "volume"]
    bundle = _load_bundle_from_args(args, features)
    return train_multivariate_multistep_model(
        bundle=bundle,
        k=k,
        cell=args.cell,
        layer_sizes=parse_layer_sizes(args.layers) or (128, 64),
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        loss=args.loss,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

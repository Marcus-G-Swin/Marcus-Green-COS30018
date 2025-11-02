""" stock prediction v0.6

what this file does at a high level:
  pulls together task 2 data processing (windowing, scaling, splitting)
  optional task 3 visualizations (candlesticks, boxplots)
  task 4 model factory (rnn stacks with lstm, gru, or simplernn)
  task 5 training flows:
    - multistep prediction (k step ahead)
    - multivariate prediction (ohlcv as inputs)
    - multivariate + multistep
  task 6 shim so you can run the ensemble code from c6_ensembles without touching existing flows """

# --- task 6 ensemble shim (adds --task6 without altering existing flows) ---
import sys as _sys
try:
    # if c6_ensembles is available we call into it
    from C6_Ensembles import run_task6
except Exception as _e:
    # leave a safe default. we will raise at runtime if user asks for task6
    run_task6 = None

def _maybe_run_task6():
    """
    tiny guard that inspects argv early.
    if user passed --task6 we fully hand off to the task 6 runner and then exit.
    keeping it here means we do not have to refactor the rest of the driver.
    """
    if "--task6" not in _sys.argv:
        return
    if run_task6 is None:
        raise RuntimeError("task 6 selected, but C6_Ensembles.run_task6 is unavailable.")

    import argparse as _argparse
    t6 = _argparse.ArgumentParser(add_help=True, description="task 6 ensemble runner")
    t6.add_argument("--task6", action="store_true", help="run the task 6 arima + dl ensemble flow and exit.")
    t6.add_argument("--ticker", type=str, default="CBA.AX")
    t6.add_argument("--ticket", type=str, default=None, help="typo safety. use --ticker")

    t6.add_argument("--start-date", type=str, default="2020-01-01")
    t6.add_argument("--test-start", type=str, default="2023-08-02")
    t6.add_argument("--end-date", type=str, default="2024-07-02")
    t6.add_argument("--n-steps", type=int, default=60)
    t6.add_argument("--multivariate", action="store_true")

    # dl options
    t6.add_argument("--cell", type=str, default="LSTM", choices=["LSTM", "GRU", "RNN"])
    t6.add_argument("--layers", type=str, default="128,64")
    t6.add_argument("--dropout", type=str, default="0.3")
    t6.add_argument("--epochs", type=int, default=20)
    t6.add_argument("--batch-size", type=int, default=32)

    # classical leg
    t6.add_argument("--arima-auto", action="store_true")
    t6.add_argument("--arima-order", type=str, default="1,1,1")
    t6.add_argument("--seasonal", action="store_true")
    t6.add_argument("--seasonal-m", type=int, default=1)
    t6.add_argument("--seasonal-order", type=str, default="0,0,0,0")
    t6.add_argument("--ma-fallback-window", type=int, default=8)

    # ensemble choices
    t6.add_argument(
        "--ensemble",
        type=str,
        default="simple_avg",
        choices=[
            "simple_avg",
            "val_weighted",
            "val_grid",
            "stacking_ridge",
            "stacking_lr_pos",
            "residual_boost",
            "auto",
        ],
    )
    t6.add_argument("--ridge-alpha", type=float, default=1.0)

    args, unknown = t6.parse_known_args()

    if args.ticket and not args.ticker:
        args.ticker = args.ticket

    from C4_ModelMaker import parse_layer_sizes
    if any(ch in args.dropout for ch in (",", " ")):
        dval = [float(x) for x in args.dropout.replace(" ", ",").split(",") if x.strip()]
    else:
        dval = float(args.dropout)
    layers = parse_layer_sizes(args.layers) or [128, 64]

    arima_order = tuple(int(x) for x in args.arima_order.replace(" ", ",").split(",") if x.strip())
    seas_order = tuple(int(x) for x in args.seasonal_order.replace(" ", ",").split(",") if x.strip())

    run_task6(
        ticker=args.ticker,
        start_date=args.start_date,
        test_start=args.test_start,
        end_date=args.end_date,
        n_steps=args.n_steps,
        use_multivariate=bool(args.multivariate),
        dl_cell=args.cell,
        dl_layers=layers,
        dl_dropout=dval,
        dl_epochs=args.epochs,
        dl_batch=args.batch_size,
        arima_auto=bool(args.arima_auto),
        arima_order=arima_order,
        seasonal=bool(args.seasonal),
        seasonal_m=args.seasonal_m,
        seasonal_order=seas_order,
        ensemble_kind=args.ensemble,
        ridge_alpha=args.ridge_alpha,
        ma_fallback_window=args.ma_fallback_window,
    )
    raise SystemExit(0)

_maybe_run_task6()
# --- end task 6 shim ---


import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tfs
from sklearn.metrics import mean_absolute_error

# task 2 preprocessing and splitting
from C2_DataProcessing import load_and_process_data

# task 3 visualizations
from C3_Visualization import (
    plot_candlesticks,
    plot_boxplot_moving_window,
)

# task 4 model factory
from C4_ModelMaker import create_dl_model, parse_layer_sizes

# task 5 training helpers
from C5_MLTasks import (
    train_multistep_model,
    train_multivariate_one_step,
    train_multivariate_multistep,
)


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _eval_mae_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def _load_bundle(
    ticker: str,
    start_date: str,
    end_date: str,
    n_steps: int,
    lookup_step: int,
    feature_columns,
    split_method: str,
    split_date: str = None,
    shuffle: bool = False,
    cache_dir: str = "./cache",
    prefer_cache: bool = True,
    force_refresh: bool = False,
    scale: bool = True,
    scaler_type: str = "minmax",
    save_scalers_path: str = "./cache/scalers.pkl",
    seed: int = 42,
):
    return load_and_process_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        n_steps=n_steps,
        lookup_step=lookup_step,
        feature_columns=feature_columns,
        nan_strategy="ffill",
        split_method=split_method,
        split_date=split_date,
        shuffle=shuffle,
        cache_dir=cache_dir,
        prefer_cache=prefer_cache,
        force_refresh=force_refresh,
        scale=scale,
        scaler_type=scaler_type,
        save_scalers_path=save_scalers_path,
        seed=seed,
    )


def _train_plot_baseline(args):
    """One step baseline that also handles your plotting flags."""
    features = ["adjclose"]
    bundle = _load_bundle(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        n_steps=args.n_steps,
        lookup_step=args.lookup_step,
        feature_columns=features,
        split_method="date",
        split_date=args.test_start,
        shuffle=False,
        cache_dir=args.cache_dir,
        prefer_cache=True,
        force_refresh=args.force_refresh,
        scale=True,
        scaler_type=args.scaler_type,
        save_scalers_path=f"{args.cache_dir}/{args.ticker}_scalers.pkl",
        seed=args.seed,
    )

    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    test_df = bundle["test_df"]
    scaler = bundle["column_scaler"]["adjclose"]

    # Build DL model
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

    # Train first, then we will plot
    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # Inference and inverse scale
    preds = model.predict(X_test, verbose=0)
    y_true = scaler.inverse_transform(_as_2d(y_test)).ravel()
    y_pred = scaler.inverse_transform(_as_2d(preds)).ravel()
    mae, rmse = _eval_mae_rmse(y_true, y_pred)
    print(f"one step test MAE: {mae:.4f}  RMSE: {rmse:.4f}")

    # If plotting flags are set, create the three windows and show once
    if args.plot_candles or args.plot_box:
        # Figure 1 - line chart
        fig_line = plt.figure(figsize=(10, 5))
        plt.plot(y_true, label=f"Actual {args.ticker} Price")
        plt.plot(y_pred, label=f"Predicted {args.ticker} Price")
        plt.title(f"{args.ticker} Share Price - test period")
        plt.xlabel("Time")
        plt.ylabel(f"{args.ticker} Share Price")
        plt.legend()
        plt.tight_layout()
        try:
            fig_line.canvas.manager.set_window_title("Line Chart")
        except Exception:
            pass

        # Build test OHLCV frame for candlesticks and boxplot
        ohlcv_test = pd.DataFrame({
            "open":   test_df["open"].values.ravel(),
            "high":   test_df["high"].values.ravel(),
            "low":    test_df["low"].values.ravel(),
            "close":  test_df["close"].values.ravel(),
            "volume": test_df["volume"].values.ravel(),
        }, index=pd.DatetimeIndex(test_df.index).sort_values())

        # Figure 2 - candlesticks
        if args.plot_candles:
            fig_candle, _ = plot_candlesticks(
                ohlcv_test,
                n=max(1, int(args.candle_n)),
                style="yahoo",
                title=f"{args.ticker} candlesticks - {max(1, int(args.candle_n))}-day blocks",
                mav=[5, 20],
                volume=False,
                tight_layout=True,
                returnfig=True,
            )
            try:
                fig_candle.canvas.manager.set_window_title("Candlesticks")
            except Exception:
                pass

        # Figure 3 - boxplot
        if args.plot_box:
            safe_metric = args.box_metric if args.box_metric in {"close", "returns", "range", "hl_pct"} else "returns"
            fig_box = plt.figure(figsize=(10, 5))
            plot_boxplot_moving_window(
                ohlcv_test,
                n=max(1, int(args.box_n)),
                metric=safe_metric,
                title=f"{args.ticker} {safe_metric} distribution - {max(1, int(args.box_n))}-day blocks",
                label_every=2,
                make_new_figure=False,
            )
            try:
                fig_box.canvas.manager.set_window_title("Boxplot")
            except Exception:
                pass

        plt.show()
    else:
        print("plots were not requested. pass --plot-candles or --plot-box to visualize.")


def _train_multistep(args):
    train_multistep_k(args)


def _train_multivariate(args):
    train_multivariate_one_step(args)


def _train_multi_multi(args):
    train_multivariate_multistep(args)


def main():
    p = argparse.ArgumentParser(description="stock prediction v0.6 driver")

    # data
    p.add_argument("--ticker", type=str, default="CBA.AX")
    p.add_argument("--start-date", type=str, default="2020-01-01")
    p.add_argument("--test-start", type=str, default="2023-08-02")
    p.add_argument("--end-date", type=str, default="2024-07-02")
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--lookup-step", type=int, default=1)
    p.add_argument("--scaler-type", type=str, default="minmax", choices=["minmax", "standard"])
    p.add_argument("--cache-dir", type=str, default="./cache")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # visual flags - your old CLI
    p.add_argument("--plot-candles", action="store_true")
    p.add_argument("--plot-box", action="store_true")
    p.add_argument("--candle-n", type=int, default=1, help="aggregate this many trading days per candlestick")
    p.add_argument("--box-n", type=int, default=5, help="days per box in the boxplot")
    p.add_argument("--box-metric", type=str, default="returns", choices=["close", "returns", "range", "hl_pct"])

    # model
    p.add_argument("--cell", type=str, default="LSTM", choices=["LSTM", "GRU", "RNN"])
    p.add_argument("--layers", type=str, default="128,64")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--loss", type=str, default="mean_absolute_error")
    p.add_argument("--optimizer", type=str, default="rmsprop")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)

    # modes
    p.add_argument("--mode", type=str, default="baseline",
                   choices=["baseline", "multistep", "multivariate", "multi-multi"])

    args = p.parse_args()

    # route to the chosen training mode
    if args.mode == "baseline":
        _train_plot_baseline(args)
    elif args.mode == "multistep":
        _train_multistep(args)
    elif args.mode == "multivariate":
        _train_multivariate(args)
    elif args.mode == "multi-multi":
        _train_multi_multi(args)
    else:
        raise ValueError("unknown mode")

if __name__ == "__main__":
    main()

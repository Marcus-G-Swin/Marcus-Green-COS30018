""" c6_ensembles.py : cos30018 option c task 6 (lean)

what this file does:
  - builds a classical leg (arima or fallback) and a dl leg (lstm by default)
  - blends them using a few strategies, including an auto chooser
  - prints metrics in price units """

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from C2_DataProcessing import load_and_process_data
from C4_ModelMaker import create_dl_model, parse_layer_sizes

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# optional arima backends. if none are available we fall back to moving average
_HAVE_PMD = False
try:
    from pmdarima import auto_arima  # type: ignore
    _HAVE_PMD = True
except Exception:
    _HAVE_PMD = False

try:
    import statsmodels.api as sm  # type: ignore
except Exception:
    sm = None  # type: ignore


# ------------- tiny utils -------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def hstack_cols(*cols: np.ndarray) -> np.ndarray:
    return np.hstack([to_2d(c) for c in cols])

def inv_scale(col_scaler, z: np.ndarray) -> np.ndarray:
    # models work in scaled space. reports need price units. this inverts cleanly.
    return col_scaler.inverse_transform(to_2d(np.asarray(z).ravel())).ravel()

@dataclass
class SplitData:
    bundle: Dict[str, object]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_df: pd.DataFrame
    scaler: object

def temporal_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2) -> Tuple[np.ndarray, ...]:
    n = len(X)
    cut = int(round((1.0 - val_frac) * n))
    return X[:cut], y[:cut], X[cut:], y[cut:]


# ------------- classical leg -------------
class _MAPredictor:
    # minimal moving average fallback so the task still runs without arima libs
    def __init__(self, series: np.ndarray, window: int = 8):
        self._last_ma = float(np.mean(np.asarray(series, float)[-max(2, int(window)):]))

    def predict(self, start_i: int, end_i: int) -> np.ndarray:
        n = end_i - start_i + 1
        return np.full(n, self._last_ma, float)

def fit_arima_series(
    series: pd.Series,
    seasonal: bool = False,
    m: int = 1,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    use_auto: bool = True,
    ma_fallback_window: int = 8,
):
    """ fit arima or sarima. returns (model_like, predict_fn) so callers do not care about backend. """
    y = series.astype(float)

    if use_auto and _HAVE_PMD:
        model = auto_arima(y, seasonal=seasonal, m=m, error_action="ignore", suppress_warnings=True, stepwise=True)
        def _predict(si: int, ei: int) -> np.ndarray:
            return np.asarray(model.predict(n_periods=ei - si + 1), float)
        return model, _predict

    if sm is not None:
        if seasonal and seasonal_order != (0, 0, 0, 0):
            model = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=seasonal_order, trend="n")
        else:
            model = sm.tsa.statespace.SARIMAX(y, order=order, trend="n")
        fit_res = model.fit(disp=False)
        def _predict(si: int, ei: int) -> np.ndarray:
            return np.asarray(fit_res.get_prediction(start=si, end=ei).predicted_mean, float)
        return fit_res, _predict

    ma_model = _MAPredictor(y.values, window=ma_fallback_window)
    print("[task6] warning: arima libraries not found. using moving average fallback.")
    return ma_model, ma_model.predict


# ------------- dl leg -------------
def train_dl_one_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    cell: str = "LSTM",
    layer_sizes: List[int] = (128, 64),
    dropout: float | List[float] = 0.3,
    bidirectional: bool = False,
    loss: str = "mean_absolute_error",
    optimizer: str = "rmsprop",
    epochs: int = 20,
    batch_size: int = 32,
) -> Tuple[object, np.ndarray]:
    # metrics=("mae",) keeps keras logs aligned with the report
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
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, model.predict(X_eval, verbose=0)


# ------------- ensemble strategies -------------
def ens_simple(a, b):  # simple average in scaled space
    return (to_2d(a) + to_2d(b)) / 2.0

def ens_val_weighted(a_val, b_val, y_val, a_test, b_test):
    mae_a = mean_absolute_error(y_val, a_val) + 1e-9
    mae_b = mean_absolute_error(y_val, b_val) + 1e-9
    wa, wb = (1.0 / mae_a), (1.0 / mae_b)
    s = wa + wb
    wa, wb = wa / s, wb / s
    return wa * to_2d(a_test) + wb * to_2d(b_test), (wa, wb)

def ens_val_grid(a_val, b_val, y_val, a_test, b_test, steps=201):
    yv, av, bv = to_2d(y_val).ravel(), to_2d(a_val).ravel(), to_2d(b_val).ravel()
    w_best, m_best = 0.5, float("inf")
    for w in np.linspace(0.0, 1.0, steps):
        m = mean_absolute_error(yv, w * av + (1 - w) * bv)
        if m < m_best:
            w_best, m_best = float(w), m
    return w_best * to_2d(a_test) + (1 - w_best) * to_2d(b_test), w_best

def ens_stack_ridge(a_val, b_val, y_val, a_test, b_test, alpha=1.0):
    Xv, Xt = hstack_cols(a_val, b_val), hstack_cols(a_test, b_test)
    meta = Ridge(alpha=alpha).fit(Xv, np.asarray(y_val).ravel())
    return to_2d(meta.predict(Xt)), np.asarray(meta.coef_, float)

def ens_stack_lr_pos(a_val, b_val, y_val, a_test, b_test):
    Xv, Xt = hstack_cols(a_val, b_val), hstack_cols(a_test, b_test)
    meta = LinearRegression(positive=True).fit(Xv, np.asarray(y_val).ravel())
    coefs = (float(meta.coef_[0]), float(meta.coef_[1]), float(getattr(meta, "intercept_", 0.0)))
    return to_2d(meta.predict(Xt)), coefs

def ens_residual_boost(arima_val, y_val, arima_test, X_val, X_test, cell="LSTM", epochs=15, batch=32):
    # dl learns residuals of the classical leg. final = arima + correction
    from C4_ModelMaker import create_dl_model
    residuals = to_2d(y_val) - to_2d(arima_val)
    model = create_dl_model(
        sequence_length=X_val.shape[1], n_features=X_val.shape[2],
        layer_type=cell, layer_sizes=[64], dropout=0.2,
        bidirectional=False, loss="mean_absolute_error", optimizer="rmsprop", metrics=("mae",),
    )
    model.fit(X_val, residuals, epochs=epochs, batch_size=batch, verbose=1)
    corr = model.predict(X_test, verbose=0)
    return to_2d(arima_test) + corr, model


# ------------- data prep -------------
def prepare_data(
    ticker: str, start_date: str, test_start: str, end_date: str,
    n_steps: int = 60, features: List[str] = ["adjclose"], scaler_type: str = "minmax",
) -> SplitData:
    bundle = load_and_process_data(
        ticker=ticker, start_date=start_date, end_date=end_date,
        n_steps=n_steps, lookup_step=1, feature_columns=features,
        nan_strategy="ffill", split_method="date", split_date=test_start,
        shuffle=False, cache_dir="./cache", prefer_cache=True, force_refresh=True,
        scale=True, scaler_type=scaler_type, save_scalers_path=f"./cache/{ticker}_scalers_task6.pkl", seed=314,
    )
    X, y, Xt, yt = bundle["X_train"], bundle["y_train"], bundle["X_test"], bundle["y_test"]
    X_tr, y_tr, X_val, y_val = temporal_val_split(X, y, 0.2)
    return SplitData(
        bundle=bundle, X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
        X_test=Xt, y_test=yt, test_df=bundle["test_df"], scaler=bundle["column_scaler"]["adjclose"],
    )


# ------------- orchestration -------------
def run_task6(
    ticker: str = "CBA.AX",
    start_date: str = "2020-01-01",
    test_start: str = "2023-08-02",
    end_date: str = "2024-07-02",
    n_steps: int = 60,
    use_multivariate: bool = False,
    dl_cell: str = "LSTM",
    dl_layers: List[int] = (128, 64),
    dl_dropout: float | List[float] = 0.3,
    dl_epochs: int = 20,
    dl_batch: int = 32,
    arima_auto: bool = True,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    seasonal: bool = False,
    seasonal_m: int = 1,
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ensemble_kind: str = "simple_avg",
    ridge_alpha: float = 1.0,
    ma_fallback_window: int = 8,
):
    # choose feature set
    feats = ["adjclose", "volume", "open", "high", "low"] if use_multivariate else ["adjclose"]
    data = prepare_data(ticker, start_date, test_start, end_date, n_steps, feats)

    # build a price-space target for classical training so it predicts native units
    y_all_scaled = np.concatenate([data.y_train, data.y_val, data.y_test], axis=0).ravel()
    scaler = data.scaler
    y_all_price = inv_scale(scaler, y_all_scaled)

    n_tr, n_val, n_te = len(data.y_train), len(data.y_val), len(data.y_test)

    # classical for validation: fit on train, forecast next n_val points
    y_price_train = y_all_price[:n_tr]
    model_tr, predict_tr = fit_arima_series(
        pd.Series(y_price_train), seasonal=seasonal, m=seasonal_m,
        order=arima_order, seasonal_order=seasonal_order, use_auto=arima_auto,
        ma_fallback_window=ma_fallback_window,
    )
    arima_val_pred_price = predict_tr(n_tr, n_tr + n_val - 1)

    # classical for test: refit on train + actual val, then forecast test
    y_val_price_actual = inv_scale(scaler, data.y_val)
    y_trainval_price = np.concatenate([y_price_train, y_val_price_actual])
    model_trv, predict_trv = fit_arima_series(
        pd.Series(y_trainval_price), seasonal=seasonal, m=seasonal_m,
        order=arima_order, seasonal_order=seasonal_order, use_auto=arima_auto,
        ma_fallback_window=ma_fallback_window,
    )
    arima_test_pred_price = predict_trv(len(y_trainval_price), len(y_trainval_price) + n_te - 1)

    # scale the classical preds so they live in the same space as dl outputs
    arima_val_pred = scaler.transform(to_2d(arima_val_pred_price))
    arima_test_pred = scaler.transform(to_2d(arima_test_pred_price))

    # dl leg
    dl_model, dl_val_pred = train_dl_one_step(
        data.X_train, data.y_train, data.X_val,
        cell=dl_cell, layer_sizes=dl_layers, dropout=dl_dropout,
        epochs=dl_epochs, batch_size=dl_batch,
    )
    dl_test_pred = dl_model.predict(data.X_test, verbose=0)

    # registry to keep code compact
    Strategy = Callable[..., Tuple[np.ndarray, object]]
    strategies: Dict[str, Strategy] = {
        "simple_avg": lambda **kw: (ens_simple(kw["a_test"], kw["b_test"]), None),
        "val_weighted": lambda **kw: ens_val_weighted(kw["a_val"], kw["b_val"], kw["y_val"], kw["a_test"], kw["b_test"]),
        "val_grid": lambda **kw: ens_val_grid(kw["a_val"], kw["b_val"], kw["y_val"], kw["a_test"], kw["b_test"]),
        "stacking_ridge": lambda **kw: ens_stack_ridge(kw["a_val"], kw["b_val"], kw["y_val"], kw["a_test"], kw["b_test"], alpha=kw.get("ridge_alpha", 1.0)),
        "stacking_lr_pos": lambda **kw: ens_stack_lr_pos(kw["a_val"], kw["b_val"], kw["y_val"], kw["a_test"], kw["b_test"]),
        "residual_boost": lambda **kw: ens_residual_boost(kw["a_val"], kw["y_val"], kw["a_test"], kw["X_val"], kw["X_test"], cell=kw.get("dl_cell", "LSTM"), epochs=max(10, kw.get("dl_epochs", 20)//2), batch=kw.get("dl_batch", 32)),
    }

    # pick a strategy
    kw = dict(
        a_val=arima_val_pred, b_val=dl_val_pred, y_val=data.y_val,
        a_test=arima_test_pred, b_test=dl_test_pred,
        X_val=data.X_val, X_test=data.X_test,
        ridge_alpha=ridge_alpha, dl_cell=dl_cell, dl_epochs=dl_epochs, dl_batch=dl_batch,
    )

    if ensemble_kind == "auto":
        # score a few on validation and choose lowest mae in scaled space
        cand = {
            "simple_avg": ens_simple(arima_val_pred, dl_val_pred),
            "val_weighted": ens_val_weighted(arima_val_pred, dl_val_pred, data.y_val, arima_val_pred, dl_val_pred)[0],
            "val_grid": ens_val_grid(arima_val_pred, dl_val_pred, data.y_val, arima_val_pred, dl_val_pred)[0],
        }
        Xv = hstack_cols(arima_val_pred, dl_val_pred)
        cand["stacking_ridge"] = to_2d(Ridge(alpha=ridge_alpha).fit(Xv, data.y_val.ravel()).predict(Xv))
        cand["stacking_lr_pos"] = to_2d(LinearRegression(positive=True).fit(Xv, data.y_val.ravel()).predict(Xv))
        cand["dl_only"] = to_2d(dl_val_pred)
        cand["classical_only"] = to_2d(arima_val_pred)
        scores = {k: mean_absolute_error(data.y_val, v) for k, v in cand.items()}
        best = min(scores, key=scores.get)
        print(f"\n[auto] picked: {best}  val mae {scores[best]:.6f}")
        if best in strategies:
            ens_test_scaled, aux = strategies[best](**kw)
        elif best == "dl_only":
            ens_test_scaled, aux = to_2d(dl_test_pred), None
        else:
            ens_test_scaled, aux = to_2d(arima_test_pred), None
        ens_val_scaled = cand[best]
        extra = aux
    else:
        ens_test_scaled, extra = strategies[ensemble_kind](**kw)
        # for validation reporting we show the simple average to keep prints short
        ens_val_scaled = ens_simple(arima_val_pred, dl_val_pred)

    # invert to price units
    y_val_price = inv_scale(scaler, data.y_val)
    y_test_price = inv_scale(scaler, data.y_test)
    dl_val_price = inv_scale(scaler, dl_val_pred)
    dl_test_price = inv_scale(scaler, dl_test_pred)
    ens_val_price = inv_scale(scaler, ens_val_scaled)
    ens_test_price = inv_scale(scaler, ens_test_scaled)
    arima_val_pred_price = inv_scale(scaler, arima_val_pred)
    arima_test_pred_price = inv_scale(scaler, arima_test_pred)

    # quick diagnostic. if high and positive, legs make similar mistakes
    res_c = y_val_price - arima_val_pred_price
    res_d = y_val_price - dl_val_price
    if len(res_c) > 1 and np.all(np.isfinite(res_c)) and np.all(np.isfinite(res_d)):
        print(f"\nresidual correlation (validation): {float(np.corrcoef(res_c, res_d)[0,1]):.3f}")

    def report(tag, yt, yp):
        print(f"{tag:>12} | MAE: {mean_absolute_error(yt, yp):8.4f}  RMSE: {rmse(yt, yp):8.4f}")

    print("\nvalidation metrics (price units)")
    report("classical", y_val_price, arima_val_pred_price)
    report("dl", y_val_price, dl_val_price)
    report("ensemble", y_val_price, ens_val_price)

    print("\ntest metrics (price units)")
    report("classical", y_test_price, arima_test_pred_price)
    report("dl", y_test_price, dl_test_price)
    report("ensemble", y_test_price, ens_test_price)

    if isinstance(extra, tuple):
        print(f"\nmeta or weights: {extra}")

    return {
        "y_val_price": y_val_price,
        "y_test_price": y_test_price,
        "classical_val_price": arima_val_pred_price,
        "classical_test_price": arima_test_pred_price,
        "dl_val_price": dl_val_price,
        "dl_test_price": dl_test_price,
        "ens_val_price": ens_val_price,
        "ens_test_price": ens_test_price,
        "weights_or_coefs": extra,
    }


# ------------- cli -------------
def main():
    p = argparse.ArgumentParser(description="task 6 ensembles (classical + dl)")
    p.add_argument("--ticker", type=str, default="CBA.AX")
    p.add_argument("--ticket", type=str, default=None)

    p.add_argument("--start-date", type=str, default="2020-01-01")
    p.add_argument("--test-start", type=str, default="2023-08-02")
    p.add_argument("--end-date", type=str, default="2024-07-02")
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--multivariate", action="store_true")

    # dl
    p.add_argument("--cell", type=str, default="LSTM", choices=["LSTM", "GRU", "RNN"])
    p.add_argument("--layers", type=str, default="128,64")
    p.add_argument("--dropout", type=str, default="0.3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)

    # classical
    p.add_argument("--arima-auto", action="store_true")
    p.add_argument("--arima-order", type=str, default="1,1,1")
    p.add_argument("--seasonal", action="store_true")
    p.add_argument("--seasonal-m", type=int, default=1)
    p.add_argument("--seasonal-order", type=str, default="0,0,0,0")
    p.add_argument("--ma-fallback-window", type=int, default=8)

    # ensemble
    p.add_argument("--ensemble", type=str, default="simple_avg",
                   choices=["simple_avg", "val_weighted", "val_grid", "stacking_ridge", "stacking_lr_pos", "residual_boost", "auto"])
    p.add_argument("--ridge-alpha", type=float, default=1.0)

    args = p.parse_args()
    if args.ticket and not args.ticker:
        args.ticker = args.ticket

    layers = parse_layer_sizes(args.layers) or [128, 64]
    dropout = [float(x) for x in args.dropout.replace(" ", ",").split(",") if x.strip()] if ("," in args.dropout or " " in args.dropout) else float(args.dropout)
    arima_order = tuple(int(x) for x in args.arima_order.replace(" ", ",").split(",") if x.strip())
    seas_order  = tuple(int(x) for x in args.seasonal_order.replace(" ", ",").split(",") if x.strip())
    if len(arima_order) != 3: raise ValueError("--arima-order must be p,d,q")
    if len(seas_order)  != 4: raise ValueError("--seasonal-order must be P,D,Q,m")

    run_task6(
        ticker=args.ticker, start_date=args.start_date, test_start=args.test_start, end_date=args.end_date,
        n_steps=args.n_steps, use_multivariate=bool(args.multivariate),
        dl_cell=args.cell, dl_layers=layers, dl_dropout=dropout,
        dl_epochs=args.epochs, dl_batch=args.batch_size,
        arima_auto=bool(args.arima_auto), arima_order=arima_order,
        seasonal=bool(args.seasonal), seasonal_m=args.seasonal_m, seasonal_order=seas_order,
        ensemble_kind=args.ensemble, ridge_alpha=args.ridge_alpha, ma_fallback_window=args.ma_fallback_window,
    )

if __name__ == "__main__":
    main()

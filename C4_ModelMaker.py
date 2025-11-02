from typing import Iterable, List, Optional, Sequence, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    SimpleRNN,
    Dense,
    Dropout,
    Bidirectional,
)

# map short string names to their actual keras layer classes.
# this lets you swap between LSTM, GRU, or vanilla RNN from command line args.
_CELL_REGISTRY = {
    "LSTM": LSTM,
    "GRU": GRU,
    "RNN": SimpleRNN,
}


def create_dl_model(
    sequence_length: int,
    n_features: int,
    layer_type: str = "LSTM",
    layer_sizes: Sequence[int] = (128, 64),
    dropout: Union[float, Sequence[float]] = 0.3,
    bidirectional: bool = False,
    loss: str = "mean_absolute_error",
    optimizer: str = "rmsprop",
    metrics: Optional[Sequence[str]] = ("mean_absolute_error",),
):
    """ builds a recurrent neural net for one-step regression.
    you can switch between lstm, gru, or simple rnn with a flag.
    the network outputs one continuous value (ie a stock price). """
    # check that the chosen cell type is valid
    lt = layer_type.strip().upper()
    if lt not in _CELL_REGISTRY:
        raise ValueError(f"Unsupported layer_type '{layer_type}'. Use one of {list(_CELL_REGISTRY)}")
    cell_cls = _CELL_REGISTRY[lt]  # e.g. "LSTM" → keras.layers.LSTM

    # make sure the layer sizes are clean integers
    if not isinstance(layer_sizes, (list, tuple)) or len(layer_sizes) == 0:
        raise ValueError("layer_sizes must be a non-empty list or tuple of ints")
    sizes: List[int] = [int(u) for u in layer_sizes]

    # normalize dropout to a list, even if only one value was passed
    # this way, every layer gets a defined dropout rate
    if isinstance(dropout, (list, tuple)):
        if len(dropout) != len(sizes):
            raise ValueError("If dropout is a list/tuple, it must match the number of layers")
        dvals = [float(d) for d in dropout]
    else:
        dvals = [float(dropout)] * len(sizes)

    # build the model layer by layer
    model = Sequential()
    for i, (units, d) in enumerate(zip(sizes, dvals)):
        # if we’re stacking multiple rnn layers, we need return_sequences=True
        # so the next layer still receives a full sequence instead of a single output vector
        return_seq = i < len(sizes) - 1
        kwargs = {"units": units, "return_sequences": return_seq}

        # the first layer needs the input shape explicitly
        if i == 0:
            kwargs["input_shape"] = (sequence_length, n_features)

        # make the recurrent layer
        rnn = cell_cls(**kwargs)

        # if bidirectional is on, wrap the rnn so it reads both forward and backward
        layer = Bidirectional(rnn) if bidirectional else rnn
        model.add(layer)

        # apply dropout between rnn layers to reduce overfitting
        if d and d > 0:
            model.add(Dropout(d))

    # final dense layer gives one numeric output per sequence
    # linear activation is used because we’re predicting continuous values
    model.add(Dense(1, activation="linear"))

    # compile model to attach training setup
    model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics or []))
    return model


def parse_layer_sizes(s: str) -> List[int]:
    """ turns a string like "128,64,32" or "256 128" into [128, 64, 32].
    helpful for reading command line inputs into actual lists. """
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    return [int(p) for p in parts]

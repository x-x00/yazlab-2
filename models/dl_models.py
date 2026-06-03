import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from config import CFG

log = logging.getLogger(__name__)


def set_seed(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_lstm(input_shape: tuple, seed: int = 42) -> keras.Model:
    set_seed(seed)
    inp = keras.Input(shape=input_shape)
    x = layers.LSTM(CFG.dl.lstm_units, return_sequences=True)(inp)
    x = layers.Dropout(CFG.dl.dropout)(x)
    x = layers.LSTM(CFG.dl.lstm_units // 2)(x)
    x = layers.Dropout(CFG.dl.dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.dl.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru(input_shape: tuple, seed: int = 42) -> keras.Model:
    set_seed(seed)
    inp = keras.Input(shape=input_shape)
    x = layers.GRU(CFG.dl.gru_units, return_sequences=True)(inp)
    x = layers.Dropout(CFG.dl.dropout)(x)
    x = layers.GRU(CFG.dl.gru_units // 2)(x)
    x = layers.Dropout(CFG.dl.dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="GRU")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.dl.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(seed: int) -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG.dl.patience,
            restore_best_weights=True
        )
    ]


def train_model(model: keras.Model,
                X_train, y_train,
                X_val,   y_val,
                seed: int = 42) -> dict:
    pos = y_train.sum()
    neg = len(y_train) - pos

    ratio = float(neg) / max(pos, 1)
    class_weight = {0: 1.0, 1: max(ratio, 5.0)}

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CFG.dl.epochs,
        batch_size=CFG.dl.batch_size,
        callbacks=get_callbacks(seed),
        class_weight=class_weight,
        verbose=0
    )
    return hist.history


def best_threshold(model: keras.Model, X_val, y_val) -> float:
    from sklearn.metrics import f1_score
    proba = model.predict(X_val, verbose=0).flatten()
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_val, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate_model(model: keras.Model,
                   X_test, y_test,
                   threshold: float = None,
                   X_val=None, y_val=None) -> dict:
    if threshold is None and X_val is not None:
        threshold = best_threshold(model, X_val, y_val)
    elif threshold is None:
        threshold = 0.5
    proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (proba >= threshold).astype(int)
    return compute_metrics(y_test, y_pred, proba)


def compute_metrics(y_true, y_pred, proba=None) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def save_dl_model(model: keras.Model, path: str):
    # .keras format
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    log.info(f"DL model saved → {path}")


def load_dl_model(path: str) -> keras.Model:
    model = keras.models.load_model(path)
    log.info(f"DL model loaded ← {path}")
    return model


def model_path(dataset: str, model_type: str, seed: int,
               scenario: str, fold: int = None,
               base_dir: str = None) -> str:
    from config import CFG
    base = base_dir or os.path.join(CFG.results_dir, "..", "models_saved")
    fold_str = f"_fold{fold}" if fold is not None else ""
    fname = f"{model_type}_seed{seed}{fold_str}_{scenario}.keras"
    return os.path.join(base, dataset, fname)

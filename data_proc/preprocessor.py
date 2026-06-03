import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from config import CFG


# normalization
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


# pca
def fit_pca(X_train: np.ndarray, n_components: int = 1) -> PCA:
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    return pca


def apply_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X)[:, 0]


# batadal time-ordered split
def batadal_split(X: np.ndarray, y: np.ndarray):
    # 60 / 20 / 20 time-ordered split (no shuffling)
    n = len(X)
    i_train = int(n * CFG.data.batadal_train_ratio)
    i_val   = i_train + int(n * CFG.data.batadal_val_ratio)

    X_train, y_train = X[:i_train],         y[:i_train]
    X_val,   y_val   = X[i_train:i_val],    y[i_train:i_val]
    X_test,  y_test  = X[i_val:],           y[i_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# skab groupkfold generator
def skab_kfold_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      n_splits: int = 5):
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))
    return splits


# noise injection
def add_gaussian_noise(X: np.ndarray, std: float = None,
                       seed: int = 42) -> np.ndarray:
    if std is None:
        std = CFG.automata.noise_std
    rng = np.random.default_rng(seed)
    return X + rng.normal(0, std, X.shape).astype(X.dtype)


# sequence builder for dl models
def build_sequences(X: np.ndarray, y: np.ndarray,
                    seq_len: int) -> tuple:
    # create sliding-window sequences for LSTM/GRU
    n_samples, n_feat = X.shape
    if n_samples <= seq_len:
        raise ValueError(f"Not enough samples ({n_samples}) for seq_len={seq_len}")

    X_seq = np.lib.stride_tricks.sliding_window_view(
        X, (seq_len, n_feat)
    ).squeeze(axis=1)                    
    y_seq = y[seq_len - 1:]          
    return X_seq.astype(np.float32), y_seq
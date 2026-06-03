from dataclasses import dataclass, field
from typing import List, Tuple
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")


@dataclass
class DataConfig:
    # SKAB
    skab_dirs: List[str] = field(default_factory=lambda: [
        os.path.join(DATA_DIR, "valve1"),
        os.path.join(DATA_DIR, "valve2"),
    ])
    skab_target_col: str = "anomaly"
    skab_exclude_cols: List[str] = field(default_factory=lambda: [
        "datetime", "changepoint", "source_group", "source_file", "anomaly"
    ])

    # BATADAL
    batadal_files: List[str] = field(default_factory=lambda: [
        os.path.join(DATA_DIR, "BATADAL_dataset04.csv"),
    ])
    batadal_target_col: str = "ATT_FLAG"
    batadal_exclude_cols: List[str] = field(default_factory=lambda: ["DATETIME", "ATT_FLAG"])
    # ATT_FLAG -999 -> 0 (normal), 1 -> 1 (attack)
    batadal_label_map: dict = field(default_factory=lambda: {-999: 0, 0: 0, 1: 1})

    # split for BATADAL (time-ordered)
    batadal_train_ratio: float = 0.60
    batadal_val_ratio:   float = 0.20
    batadal_test_ratio:  float = 0.20


@dataclass
class PreprocessConfig:
    normalize: bool = True
    apply_pca: bool = True   # for automata model
    pca_components: int = 1


@dataclass
class AutomataConfig:
    # fixed comparison params
    window_size_fixed: int = 4
    alphabet_size_fixed: int = 3

    # variation grids
    window_sizes:   List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    alphabet_sizes: List[int] = field(default_factory=lambda: [3, 4, 5, 6])

    # gaussian noise
    noise_std: float = 0.05

    # anomaly threshold: path probabilities below this are considired anomaly
    anomaly_threshold: float = 0.05


@dataclass
class DLConfig:
    epochs: int = 50
    batch_size: int = 32
    patience: int = 5           # early stopping
    sequence_length: int = 30   # look-back window for LSTM/GRU
    lstm_units: int = 64
    gru_units: int = 64
    cnn_filters: int = 64
    cnn_kernel: int = 3
    dropout: float = 0.3
    learning_rate: float = 1e-3


@dataclass
class ExperimentConfig:
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2026, 7, 999])
    skab_n_splits: int = 5      # groupkfold


@dataclass
class Config:
    data:       DataConfig      = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    automata:   AutomataConfig  = field(default_factory=AutomataConfig)
    dl:         DLConfig        = field(default_factory=DLConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    results_dir: str = RESULTS_DIR
    plots_dir:   str = PLOTS_DIR


CFG = Config()

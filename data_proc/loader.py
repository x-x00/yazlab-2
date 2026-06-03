import os
import glob
import pandas as pd
import numpy as np
from config import CFG


def load_skab() -> pd.DataFrame:
    # load all csv files for skab data. adds source_group and source_file columns for groupkflod splitting.
    frames = []
    for folder in CFG.data.skab_dirs:
        group = os.path.basename(folder)
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        for fpath in csv_files:
            df = pd.read_csv(fpath, sep=";", parse_dates=["datetime"])
            df["source_group"] = group
            stem = os.path.splitext(os.path.basename(fpath))[0]
            df["source_file"] = f"{group}_{stem}"
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    n_anom = combined["anomaly"].sum()
    print(f"[SKAB] Loaded {len(combined):,} rows | "
          f"{int(n_anom):,} anomalies ({n_anom/len(combined):.1%}) | "
          f"{combined['source_file'].nunique()} source files "
          f"from {len(CFG.data.skab_dirs)} folders")
    return combined


def load_batadal() -> pd.DataFrame:
    # load all batadal files.  dataset04: ATT_FLAG=-999 (normal) or 1 (attack).
    frames = []
    for fpath in CFG.data.batadal_files:
        if not os.path.exists(fpath):
            print(f"[BATADAL] Warning: {fpath} not found — skipping")
            continue
        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()
        df["source_file"] = os.path.basename(fpath)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = combined.columns.str.strip()

    combined[CFG.data.batadal_target_col] = (
        combined[CFG.data.batadal_target_col]
        .map(CFG.data.batadal_label_map)
    )

    combined["DATETIME"] = pd.to_datetime(combined["DATETIME"],
                                          format="%d/%m/%y %H")
    combined = combined.sort_values("DATETIME").reset_index(drop=True)

    n_attack = combined[CFG.data.batadal_target_col].sum()
    print(f"[BATADAL] Loaded {len(combined):,} rows | "
          f"{int(n_attack):,} attacks ({n_attack/len(combined):.1%}) | "
          f"{combined['source_file'].nunique()} files")
    return combined


def get_skab_features(df: pd.DataFrame) -> tuple:
    # return X (features), y (labels), groups (for groupkfold)
    feature_cols = [c for c in df.columns if c not in CFG.data.skab_exclude_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df[CFG.data.skab_target_col].values.astype(int)
    groups = df["source_file"].values
    return X, y, groups, feature_cols


def get_batadal_features(df: pd.DataFrame) -> tuple:
    exclude = set(CFG.data.batadal_exclude_cols) | {"source_file"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(np.float32)
    y = df[CFG.data.batadal_target_col].values.astype(int)
    return X, y, feature_cols

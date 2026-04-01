
# CONFIG

DATASET_NAME = "Buchwald-Hartwig"
SPLITS_DIR   = "../data/data/split"
SPLIT_GLOB   = "dataset_1_*.npz"

OUT_CSV     = "bh_fp_ablation_all_splits_radius.csv"
SUMMARY_CSV = "bh_fp_ablation_summary_radius.csv"

TRAIN_FRAC  = 0.70
RANDOM_SEED = 42

USE_CHIRALITY = False
USE_FEATURES  = False
CLIP_COUNTS   = 10.0
FP_LOG1P      = True

GRID = [
    ("bh_r2_nb2048",  [2],    [2048]),
    ("bh_r3_nb2048",  [3],    [2048]),
    ("bh_r23_nb2048", [2, 3], [2048, 2048]),
]

MLP_KW = dict(
    hidden_layer_sizes=(1024, 512),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=1500,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
    verbose=False,
)

# END CONFIG

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils.common import (
    make_generators,
    load_bh_npz,
    rxn_to_feature_vector,
)


def split_train_test(rxn_text, y_all, train_idx, test_idx):
    N = len(y_all)

    if train_idx is not None and test_idx is not None:
        train_idx = np.array(train_idx).astype(int).reshape(-1)
        test_idx = np.array(test_idx).astype(int).reshape(-1)

        rxn_tr = [rxn_text[i] for i in train_idx]
        rxn_te = [rxn_text[i] for i in test_idx]
        y_tr = y_all[train_idx]
        y_te = y_all[test_idx]

        split_desc = f"npz indices: train={len(train_idx)} test={len(test_idx)}"
        return rxn_tr, rxn_te, y_tr, y_te, split_desc

    n_train = int(round(TRAIN_FRAC * N))
    rxn_tr, rxn_te = rxn_text[:n_train], rxn_text[n_train:]
    y_tr, y_te = y_all[:n_train], y_all[n_train:]
    split_desc = f"first {int(TRAIN_FRAC * 100)}%: train={len(rxn_tr)} test={len(rxn_te)}"
    return rxn_tr, rxn_te, y_tr, y_te, split_desc


def eval_one_split(path: Path):
    rxn_text, y_all, train_idx, test_idx = load_bh_npz(path)
    rxn_tr, rxn_te, y_tr, y_te, split_desc = split_train_test(rxn_text, y_all, train_idx, test_idx)

    print(f"\n[split] {path.name} | N={len(y_all)} | {split_desc}")

    split_rows = []
    for cfg, radii, nbits_list in GRID:
        print(f"  [cfg] {cfg} radii={radii} nBits={nbits_list}")

        gens = make_generators(radii, nbits_list, use_chirality=USE_CHIRALITY)
        cache_key_base = (
            tuple(radii),
            tuple(nbits_list),
            bool(USE_CHIRALITY),
            bool(USE_FEATURES),
            float(CLIP_COUNTS),
            bool(FP_LOG1P),
        )
        cache = {}

        t0 = time.time()
        X_tr = np.stack(
            [
                rxn_to_feature_vector(
                    r,
                    gens=gens,
                    cache_key_base=cache_key_base,
                    cache=cache,
                    use_chirality=USE_CHIRALITY,
                    use_features=USE_FEATURES,
                    fp_log1p=FP_LOG1P,
                    clip_counts=CLIP_COUNTS,
                )
                for r in rxn_tr
            ],
            axis=0,
        )
        X_te = np.stack(
            [
                rxn_to_feature_vector(
                    r,
                    gens=gens,
                    cache_key_base=cache_key_base,
                    cache=cache,
                    use_chirality=USE_CHIRALITY,
                    use_features=USE_FEATURES,
                    fp_log1p=FP_LOG1P,
                    clip_counts=CLIP_COUNTS,
                )
                for r in rxn_te
            ],
            axis=0,
        )
        feat_time = time.time() - t0

        y_mu = float(np.mean(y_tr))
        y_sd = float(np.std(y_tr) + 1e-8)
        y_tr_s = (y_tr - y_mu) / y_sd

        X_mu = X_tr.mean(axis=0)
        X_sd = X_tr.std(axis=0) + 1e-8
        X_tr_s = (X_tr - X_mu) / X_sd
        X_te_s = (X_te - X_mu) / X_sd

        t1 = time.time()
        model = MLPRegressor(**MLP_KW)
        model.fit(X_tr_s, y_tr_s)
        train_time = time.time() - t1

        yhat_te_s = model.predict(X_te_s).astype(np.float32)
        yhat_te = yhat_te_s * y_sd + y_mu

        r2 = float(r2_score(y_te, yhat_te))
        rmse = float(np.sqrt(mean_squared_error(y_te, yhat_te)))
        mae = float(mean_absolute_error(y_te, yhat_te))

        D = int(sum(nbits_list))
        xdim = int(6 * D + 6)

        print(f"    R2={r2:.6f} RMSE={rmse:.4f} MAE={mae:.4f} train={train_time:.2f}s")

        split_rows.append({
            "split_file": path.name,
            "config": cfg,
            "radii": "-".join(map(str, radii)),
            "nbits": "-".join(map(str, nbits_list)),
            "D_sum": D,
            "xdim": xdim,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "feat_time_s": feat_time,
            "train_time_s": train_time,
            "N": int(len(y_all)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "split_rule": split_desc,
        })

    return split_rows


def main():
    splits_dir = Path(SPLITS_DIR).resolve()
    files = sorted([p for p in splits_dir.glob(SPLIT_GLOB) if p.is_file()])
    if not files:
        raise SystemExit(f"No files matched: {splits_dir}/{SPLIT_GLOB}")

    print(f"[INFO] dataset: {DATASET_NAME}")
    print(f"[INFO] splits_dir: {splits_dir}")
    print(f"[INFO] found {len(files)} split files")

    all_rows = []
    for f in files:
        all_rows.extend(eval_one_split(f))

    df = pd.DataFrame(all_rows)
    out_path = Path(OUT_CSV).resolve()
    df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")

    summary = (
        df.groupby("config", as_index=False)
          .agg(
              r2_mean=("r2", "mean"),
              r2_std=("r2", "std"),
              rmse_mean=("rmse", "mean"),
              rmse_std=("rmse", "std"),
              mae_mean=("mae", "mean"),
              mae_std=("mae", "std"),
              train_time_mean=("train_time_s", "mean"),
              feat_time_mean=("feat_time_s", "mean"),
          )
          .sort_values("r2_mean", ascending=False)
    )

    sum_path = Path(SUMMARY_CSV).resolve()
    summary.to_csv(sum_path, index=False)
    print(f"[saved] {sum_path}")


if __name__ == "__main__":
    main()
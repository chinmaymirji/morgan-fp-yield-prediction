# =========================
# CONFIG
# =========================

DATASET_NAME = "Buchwald-Hartwig"
SPLITS_DIR   = "../data/data/split"
SPLIT_GLOB   = "dataset_1_*.npz"

TRAIN_N      = 2767
VAL_FRAC     = 0.10

OUT_ROOT  = "../trained_models/BH"
PLOT_DIR  = "../trained_models/BH/plots"

USE_CUDA = True
USE_AMP  = True

FP_NBITS_LIST  = [2048]
FP_RADII       = [2]
USE_CHIRALITY  = True
USE_FEATURES   = False
FP_LOG1P       = True
CLIP_COUNTS    = 50.0

SEED_BASE      = 123
EPOCHS         = 1500
MICRO_BS       = 256
EFFECTIVE_BS   = 1024
LR_PEAK        = 8e-4
WEIGHT_DECAY   = 1e-2
DROPOUT        = 0.08
EARLY_PATIENCE = 70
WARMUP_EPOCHS  = 10
MIN_LR         = 1e-5
GRAD_CLIP_NORM = 1.0

USE_MIXUP      = True
MIXUP_ALPHA    = 0.15
MIXUP_PROB     = 0.40
FEAT_DROPOUT_P = 0.03

USE_BLOCK_DROPOUT = True
BLOCK_DROPOUT_P   = 0.06

USE_EMA    = True
EMA_DECAY  = 0.999

ENSEMBLE_SEEDS = [0,1,2,3,4]

USE_WEIGHTED_ENSEMBLE = True
WEIGHT_HEAD_STEPS     = 800
WEIGHT_HEAD_LR        = 0.08
WEIGHT_HEAD_L2        = 1e-4

NUM_WORKERS = 2
PIN_MEMORY  = True
VAL_BS      = 1024
TEST_BS     = 1024

MC_PASSES_TEST = 1

HUBER_BETA = 0.7

# END CONFIG

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import json
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from utils.common import (
    set_seed,
    normalize_rxn_text,
    XYDataset,
    MLP,
    EMA,
    save_scatter,
    compute_metrics,
    apply_feat_dropout,
    get_lr,
    load_split_npz_to_df,
    make_generators,
    rxn_to_feature_vector,
)

_FP_CACHE = {}


def build_feature_matrix(rxns):
    gens = make_generators(FP_RADII, FP_NBITS_LIST, use_chirality=USE_CHIRALITY)
    cache_key_base = (
        tuple(FP_RADII),
        tuple(FP_NBITS_LIST),
        bool(USE_CHIRALITY),
        bool(USE_FEATURES),
        float(CLIP_COUNTS),
        bool(FP_LOG1P),
    )

    xdim = int(6 * sum(FP_NBITS_LIST) + 6)
    X = np.zeros((len(rxns), xdim), dtype=np.float32)

    for i, r in enumerate(rxns):
        X[i] = rxn_to_feature_vector(
            r,
            gens=gens,
            cache_key_base=cache_key_base,
            cache=_FP_CACHE,
            use_chirality=USE_CHIRALITY,
            use_features=USE_FEATURES,
            fp_log1p=FP_LOG1P,
            clip_counts=CLIP_COUNTS,
        )
    return X

def apply_mixup(x, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], lam * y + (1.0 - lam) * y[perm]


def apply_block_dropout(x, p, block_dim, nblocks=6):
    if not p or p <= 0:
        return x
    B = x.size(0)
    do = (torch.rand(B, device=x.device) < float(p))
    if not do.any():
        return x
    which = torch.randint(0, nblocks, (B,), device=x.device)
    x2 = x.clone()
    for k in range(nblocks):
        sel = do & (which == k)
        if sel.any():
            a = k * block_dim
            b = (k + 1) * block_dim
            x2[sel, a:b] = 0.0
    return x2

@torch.no_grad()
def eval_model(model, loader, y_mu, y_sig, device, use_amp, ema=None, mc_passes=1):
    if ema is not None:
        ema.apply_to(model)

    if mc_passes and mc_passes > 1:
        model.train()
    else:
        model.eval()

    ys, yh = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(use_amp and device.type == "cuda")
        ):
            if mc_passes and mc_passes > 1:
                acc = None
                for _ in range(int(mc_passes)):
                    out = model(xb)
                    acc = out if acc is None else (acc + out)
                pred_s = acc / float(mc_passes)
            else:
                pred_s = model(xb)

        pred = pred_s * y_sig + y_mu
        ys.append(yb.detach().cpu().numpy())
        yh.append(pred.detach().cpu().numpy())

    if ema is not None:
        ema.restore(model)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yh, axis=0)
    return compute_metrics(y_true, y_pred), y_true, y_pred


def fit_weighted_ensemble(y_true, pred_mat, steps=WEIGHT_HEAD_STEPS, lr=WEIGHT_HEAD_LR, l2=WEIGHT_HEAD_L2):
    y_true_t = torch.tensor(y_true, dtype=torch.float32)
    pred_t = torch.tensor(pred_mat, dtype=torch.float32)

    M = pred_t.shape[1]
    logits = torch.zeros((M,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=float(lr))

    for _ in range(int(steps)):
        w = torch.softmax(logits, dim=0)
        yhat = (pred_t * w.unsqueeze(0)).sum(dim=1)
        loss = torch.mean((yhat - y_true_t) ** 2) + float(l2) * torch.sum(w * w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    w = torch.softmax(logits.detach(), dim=0).cpu().numpy()
    return w.astype(np.float32)

def train_one_seed(split_name, outdir, seed, X_train, y_train_s, X_val, y_val, y_mu, y_sig, device):
    torch.backends.cudnn.benchmark = True

    model = MLP(in_dim=X_train.shape[1], dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))
    ema = EMA(model, EMA_DECAY) if USE_EMA else None

    micro_bs = int(MICRO_BS)
    accum_steps = max(1, int(EFFECTIVE_BS) // micro_bs)
    print(f"[INFO] batch: micro_bs={micro_bs} accum_steps={accum_steps} => effective_bs~{micro_bs*accum_steps}")

    dl_train = torch.utils.data.DataLoader(
        XYDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_s)),
        batch_size=micro_bs,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    dl_val = torch.utils.data.DataLoader(
        XYDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=int(VAL_BS),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    best_val_r2 = -1e9
    best_path = os.path.join(outdir, f"best_seed{seed}.pt")
    bad = 0

    block_dim = int(sum(FP_NBITS_LIST))

    for ep in range(1, int(EPOCHS) + 1):
        lr = get_lr(ep, EPOCHS, WARMUP_EPOCHS, LR_PEAK, MIN_LR)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []
        for step, (xb, yb) in enumerate(dl_train, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if USE_BLOCK_DROPOUT:
                xb = apply_block_dropout(xb, BLOCK_DROPOUT_P, block_dim, nblocks=6)
            xb = apply_feat_dropout(xb, FEAT_DROPOUT_P)

            if USE_MIXUP and (random.random() < MIXUP_PROB) and xb.size(0) >= 2:
                xb, yb = apply_mixup(xb, yb, alpha=MIXUP_ALPHA)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(USE_AMP and device.type == "cuda")
            ):
                pred = model(xb)
                loss = F.smooth_l1_loss(pred, yb, beta=float(HUBER_BETA)) / float(accum_steps)

            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu().item() * accum_steps))

            if step % accum_steps == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP_NORM))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        (_, _, val_r2), _, _ = eval_model(
            model,
            dl_val,
            y_mu=y_mu,
            y_sig=y_sig,
            device=device,
            use_amp=USE_AMP,
            ema=ema,
            mc_passes=1,
        )

        if ep == 1 or ep % 10 == 0:
            print(f"[{split_name} seed{seed}] ep={ep:03d} lr={lr:.2e} loss={np.mean(losses):.4f} val_r2={val_r2:.4f} best={best_val_r2:.4f}")

        if val_r2 > best_val_r2 + 1e-7:
            best_val_r2 = val_r2
            bad = 0
            os.makedirs(outdir, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "ema_state": (ema.shadow if ema is not None else None),
                "x_mean": None,
                "x_std": None,
                "y_mean": float(y_mu),
                "y_std": float(y_sig),
                "config": {
                    "x_dim": int(X_train.shape[1]),
                    "feature_layout": "6block_rxn_to_feature_vector",
                    "FP_RADII": FP_RADII,
                    "FP_NBITS_LIST": FP_NBITS_LIST,
                    "FP_LOG1P": FP_LOG1P,
                    "USE_CHIRALITY": USE_CHIRALITY,
                    "USE_FEATURES": USE_FEATURES,
                    "CLIP_COUNTS": CLIP_COUNTS,
                    "DROPOUT": DROPOUT,
                },
            }, best_path)
        else:
            bad += 1
            if bad >= int(EARLY_PATIENCE):
                print(f"[{split_name} seed{seed}] early stop @ ep={ep} (patience={EARLY_PATIENCE})")
                break

    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["state_dict"], strict=True)
    if USE_EMA and ck.get("ema_state") is not None:
        ema = EMA(model, EMA_DECAY)
        ema.shadow = ck["ema_state"]
    else:
        ema = None

    return model, ema, best_path, best_val_r2

def train_one_split(df, split_name, outdir, base_seed, device):
    df = df.copy()
    df["rxn"] = df["rxn"].astype(str).map(normalize_rxn_text)
    df["y"] = df["y"].astype(float).astype(np.float32)

    if len(df) <= TRAIN_N:
        raise RuntimeError(f"[{split_name}] dataset too small: n={len(df)} <= TRAIN_N={TRAIN_N}")

    trainval = df.iloc[:TRAIN_N].reset_index(drop=True)
    test = df.iloc[TRAIN_N:].reset_index(drop=True)

    n_tv = len(trainval)
    n_val = int(max(1, round(VAL_FRAC * n_tv)))
    n_train = n_tv - n_val

    idx = np.arange(n_tv)
    rng = np.random.RandomState(int(base_seed))
    rng.shuffle(idx)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:]

    df_train = trainval.iloc[idx_train].reset_index(drop=True)
    df_val = trainval.iloc[idx_val].reset_index(drop=True)
    df_test = test.reset_index(drop=True)

    y_train = df_train["y"].values.astype(np.float32)
    y_mu = float(np.mean(y_train))
    y_sig = float(np.std(y_train) + 1e-8)

    def y_stdize(y):
        return (y - y_mu) / y_sig

    X_train_raw = build_feature_matrix(df_train["rxn"].tolist())
    X_val_raw = build_feature_matrix(df_val["rxn"].tolist())
    X_test_raw = build_feature_matrix(df_test["rxn"].tolist())

    x_mu = X_train_raw.mean(axis=0, keepdims=True).astype(np.float32)
    x_sig = (X_train_raw.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    X_train = ((X_train_raw - x_mu) / x_sig).astype(np.float32)
    X_val = ((X_val_raw - x_mu) / x_sig).astype(np.float32)
    X_test = ((X_test_raw - x_mu) / x_sig).astype(np.float32)

    y_train_s = y_stdize(df_train["y"].values.astype(np.float32)).astype(np.float32)
    y_val = df_val["y"].values.astype(np.float32)
    y_test = df_test["y"].values.astype(np.float32)

    models, emas, val_r2s, paths = [], [], [], []

    for s in ENSEMBLE_SEEDS:
        seed = int(base_seed + 1000 * s + 17)
        set_seed(seed)
        m, ema, p, best_val_r2 = train_one_seed(
            split_name,
            outdir,
            seed,
            X_train,
            y_train_s,
            X_val,
            y_val,
            y_mu,
            y_sig,
            device
        )
        ck = torch.load(p, map_location="cpu")
        ck["x_mean"] = x_mu.squeeze(0)
        ck["x_std"] = x_sig.squeeze(0)
        torch.save(ck, p)

        models.append(m)
        emas.append(ema)
        val_r2s.append(best_val_r2)
        paths.append(p)

    dl_val = torch.utils.data.DataLoader(
        XYDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=int(VAL_BS),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    dl_test = torch.utils.data.DataLoader(
        XYDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=int(TEST_BS),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_preds = []
    for m, ema in zip(models, emas):
        (_, _, _), yv_true, yv_pred = eval_model(
            m, dl_val, y_mu=y_mu, y_sig=y_sig, device=device, use_amp=USE_AMP, ema=ema, mc_passes=1
        )
        val_preds.append(yv_pred)
    val_pred_mat = np.stack(val_preds, axis=1)

    test_preds = []
    for m, ema in zip(models, emas):
        (_, _, _), yt_true, yt_pred = eval_model(
            m,
            dl_test,
            y_mu=y_mu,
            y_sig=y_sig,
            device=device,
            use_amp=USE_AMP,
            ema=ema,
            mc_passes=int(MC_PASSES_TEST) if MC_PASSES_TEST and MC_PASSES_TEST > 1 else 1
        )
        test_preds.append(yt_pred)
    test_pred_mat = np.stack(test_preds, axis=1)

    if USE_WEIGHTED_ENSEMBLE and val_pred_mat.shape[1] >= 2:
        w = fit_weighted_ensemble(
            yv_true,
            val_pred_mat,
            steps=WEIGHT_HEAD_STEPS,
            lr=WEIGHT_HEAD_LR,
            l2=WEIGHT_HEAD_L2
        )
        y_pred_ens = (test_pred_mat * w.reshape(1, -1)).sum(axis=1)
        ensemble_mode = "weighted_softmax_valfit"
    else:
        w = np.ones((test_pred_mat.shape[1],), dtype=np.float32) / float(test_pred_mat.shape[1])
        y_pred_ens = test_pred_mat.mean(axis=1)
        ensemble_mode = "mean"

    mae, rmse, r2 = compute_metrics(yt_true, y_pred_ens)
    y_pred_ens = np.clip(y_pred_ens, 0.0, 100.0)
    mae_pct = float(mae * 100.0)
    rmse_pct = float(rmse * 100.0)

    os.makedirs(PLOT_DIR, exist_ok=True)
    scatter_path = os.path.join(PLOT_DIR, f"{split_name}_measured_vs_pred.png")
    save_scatter(yt_true, y_pred_ens, f"{DATASET_NAME} {split_name} Morgan+ANN ({ensemble_mode})", scatter_path)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            "split": split_name,
            "train_n": int(len(df_train)),
            "val_n": int(len(df_val)),
            "test_n": int(len(df_test)),
            "y_mean": y_mu,
            "y_std": y_sig,
            "x_dim": int(X_train.shape[1]),
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "val_r2_each": val_r2s,
            "ckpts": paths,
            "ensemble_mode": ensemble_mode,
            "ensemble_weights": [float(x) for x in w.tolist()],
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mae_%": float(mae_pct),
            "rmse_%": float(rmse_pct),
            "mc_passes_test": int(MC_PASSES_TEST),
            "device": str(device),
        }, f, indent=2)

    return {
        "name": split_name,
        "train_n": int(len(df_train)),
        "val_n": int(len(df_val)),
        "test_n": int(len(df_test)),
        "x_dim": int(X_train.shape[1]),
        "ensemble_mode": ensemble_mode,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mae_%": float(mae_pct),
        "rmse_%": float(rmse_pct),
        "val_r2_mean": float(np.mean(val_r2s)),
        "outdir": outdir,
        "plot": scatter_path,
    }, yt_true, y_pred_ens


def main():
    set_seed(SEED_BASE)

    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    print("[INFO] torch:", torch.__version__)
    print("[INFO] device:", device.type)
    print("[INFO] cwd:", os.getcwd())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir_abs = os.path.abspath(os.path.join(script_dir, SPLITS_DIR))
    print("[INFO] splits_dir:", SPLITS_DIR, "=>", splits_dir_abs)

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(splits_dir_abs, SPLIT_GLOB)))
    if not files:
        raise RuntimeError(f"No split files match: {os.path.join(splits_dir_abs, SPLIT_GLOB)}")

    print("[INFO] found", len(files), "split files")

    rows, all_true, all_pred = [], [], []

    for k, path in enumerate(files):
        split_name = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(
            OUT_ROOT,
            f"outputs_{DATASET_NAME.replace('–','-').replace(' ','_').lower()}_morgan_ann_{split_name}"
        )
        os.makedirs(outdir, exist_ok=True)

        print(f"\n=== [{k+1}/{len(files)}] {split_name} ===")
        df = load_split_npz_to_df(path)[["rxn", "y"]]

        if k == 0:
            print("[SANITY] example rxn strings:")
            for ex in df["rxn"].head(3).tolist():
                print("  ", ex)

        r, y_true, y_pred = train_one_split(
            df,
            split_name,
            outdir,
            base_seed=int(SEED_BASE + k),
            device=device
        )

        rows.append(r)
        all_true.append(y_true)
        all_pred.append(y_pred)

    res = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_ROOT, "results_morgan_ann.csv")
    res.to_csv(out_csv, index=False)

    yT = np.concatenate(all_true, axis=0)
    yP = np.concatenate(all_pred, axis=0)
    overall_path = os.path.join(PLOT_DIR, "ALLSPLITS_measured_vs_pred.png")
    save_scatter(yT, yP, f"{DATASET_NAME} ALL SPLITS Morgan+ANN", overall_path)

    print("\n=== Summary (mean ± std) ===")
    for kk in ["mae", "rmse", "r2", "mae_%", "rmse_%"]:
        print(f"{kk:10s} mean={res[kk].mean():.6f}  std={res[kk].std(ddof=0):.6f}")

    print("Wrote:", out_csv)
    print("Plots in:", PLOT_DIR)
    print("Overall plot:", overall_path)


if __name__ == "__main__":
    main()
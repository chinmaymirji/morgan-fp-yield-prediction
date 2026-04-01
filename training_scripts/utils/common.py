# utils/common.py

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# General shared utilities

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rxn_text(s):
    s = str(s)
    s = s.replace(" ", "")
    s = s.replace("|", "")
    s = s.replace("~", ".")
    return s


def split_rxn_3parts(r):
    parts = str(r).split(">")
    if len(parts) == 2:
        left, right = parts
        return left, "", right
    if len(parts) >= 3:
        left = parts[0]
        mid = parts[1]
        right = ">".join(parts[2:])
        return left, mid, right
    return str(r), "", ""


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in getattr(self, "backup", {}):
                p.copy_(self.backup[n])
        self.backup = {}


def save_scatter(y_true, y_pred, title, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True yield")
    plt.ylabel("Predicted yield")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def apply_feat_dropout(x, p):
    if p is None or p <= 0:
        return x
    keep = (torch.rand_like(x) > p).to(x.dtype)
    return x * keep / (1.0 - p)


def get_lr(ep, total_epochs, warmup_epochs, lr_peak, lr_min):
    if ep <= warmup_epochs:
        return lr_peak * (ep / max(1, warmup_epochs))
    t = (ep - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * t))


# Reaction-token helpers

def tokens_from_side(side: str):
    toks_ = [t for t in str(side).split(".") if t]
    toks_ = [t for t in toks_ if t.lower() != "nan"]
    return toks_


def toks(side: str):
    return tokens_from_side(side)


def rxn_to_roles(rxn):
    """
    Common reaction parsing used across SM/BH scripts.

    Returns:
        L    : tokens on left side
        M    : tokens on middle side
        prod : first product token if present else ""
    """
    rxn = normalize_rxn_text(rxn)
    left, mid, right = split_rxn_3parts(rxn)
    L = tokens_from_side(left)
    M = tokens_from_side(mid) if mid else []
    R = tokens_from_side(right)
    prod = R[0] if len(R) > 0 else ""
    return L, M, prod


def row7_to_rxn_text(row):
    """
    Convert a (7,) row like:
      [react1, react2, cat, lig, base, solv, product]
    into:
      react1.react2>cat.lig.base.solv>product
    """
    row = [normalize_rxn_text(x) for x in row]
    if len(row) < 2:
        return normalize_rxn_text(">".join(row))
    if len(row) == 2:
        left = row[0]
        prod = row[1]
        return f"{left}>>{prod}"

    prod = row[-1]
    prec = row[:-1]
    left_cols = prec[:2]
    mid_cols = prec[2:]

    left = ".".join([x for x in left_cols if x and x.lower() != "nan"])
    mid = ".".join([x for x in mid_cols if x and x.lower() != "nan"])

    if mid == "":
        return f"{left}>>{prod}"
    return f"{left}>{mid}>{prod}"


# Loading helpers

def load_bh_npz(npz_path):
    """
   Returns:
        rxn_text, y, train_idx, test_idx
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if "rxn" not in data:
        raise RuntimeError(f"Missing 'rxn' key in {npz_path}")

    rxn = data["rxn"]

    y_key = None
    for cand in ["yld", "y", "yield", "Y", "labels"]:
        if cand in data:
            y_key = cand
            break
    if y_key is None:
        raise RuntimeError(f"Missing yield key in {npz_path} (expected yld/y/yield/...)")

    y = data[y_key].astype(np.float32).reshape(-1)

    if rxn.ndim == 1:
        rxn_text = [normalize_rxn_text(x) for x in rxn.astype(str)]
    elif rxn.ndim == 2:
        rxn_text = [row7_to_rxn_text(row) for row in rxn.astype(str)]
    else:
        raise RuntimeError(f"Unsupported rxn shape: {rxn.shape}")

    train_idx = data["train_idx"] if "train_idx" in data else None
    test_idx = data["test_idx"] if "test_idx" in data else None

    return rxn_text, y, train_idx, test_idx


def load_split_npz_to_df(path):
    """
    DataFrame-oriented version of the NPZ loader.

    Returns:
        pd.DataFrame with columns ['rxn', 'y']
    """
    path = Path(path)
    rxn_text, y, _, _ = load_bh_npz(path)
    if len(rxn_text) != len(y):
        raise RuntimeError(f"[{path.name}] length mismatch: rxn={len(rxn_text)} y={len(y)}")
    return pd.DataFrame({"rxn": rxn_text, "y": y})


# Fingerprint helpers

def make_generators(radii, nbits_list, use_chirality=False):
    gens = []
    for r, nb in zip(radii, nbits_list):
        gen = GetMorganGenerator(
            radius=int(r),
            fpSize=int(nb),
            includeChirality=bool(use_chirality),
        )
        gens.append((int(r), int(nb), gen))
    return gens


def _mol_from_smiles(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def mol_morgan_count_fp(
    mol,
    radii=None,
    nbits_list=None,
    use_chirality=False,
    use_features=False,
    fp_log1p=True,
    clip_counts=None,
    gens=None,
    cache_key_base=None,
    cache=None,
):
    """
    Supports both styles:
    1) direct radii/nbits_list hashing (train scripts)
    2) prebuilt gens + cache (ablation scripts)
    """
    if gens is not None:
        D = sum(nb for _r, nb, _g in gens)
        if mol is None:
            return np.zeros((D,), dtype=np.float32)

        key = None
        if cache is not None:
            smi = Chem.MolToSmiles(mol, canonical=True)
            key = (smi,) + tuple(cache_key_base or ())
            if key in cache:
                return cache[key].copy()

        v = np.zeros((D,), dtype=np.float32)
        off = 0

        for r, nb, gen in gens:
            try:
                fp = gen.GetCountFingerprint(mol)
            except AttributeError:
                fp = AllChem.GetHashedMorganFingerprint(
                    mol,
                    radius=int(r),
                    nBits=int(nb),
                    useChirality=bool(use_chirality),
                    useFeatures=bool(use_features),
                )

            nz = fp.GetNonzeroElements()
            for idx, cnt in nz.items():
                v[off + int(idx)] = float(cnt)
            off += nb

        if clip_counts is not None and clip_counts > 0:
            np.clip(v, 0.0, float(clip_counts), out=v)

        if fp_log1p:
            v = np.log1p(v).astype(np.float32)

        if cache is not None:
            cache[key] = v
        return v.copy()

    if radii is None or nbits_list is None:
        raise ValueError("Provide either gens=... or both radii and nbits_list.")

    out_dim = int(sum(nbits_list))
    v = np.zeros((out_dim,), dtype=np.float32)
    if mol is None:
        return v

    off = 0
    for r, nb in zip(radii, nbits_list):
        fp = AllChem.GetHashedMorganFingerprint(
            mol,
            radius=int(r),
            nBits=int(nb),
            useChirality=bool(use_chirality),
            useFeatures=bool(use_features),
        )
        nz = fp.GetNonzeroElements()
        for idx, cnt in nz.items():
            if 0 <= idx < nb:
                v[off + idx] = float(cnt)
        off += nb

    if clip_counts is not None and clip_counts > 0:
        np.clip(v, 0.0, float(clip_counts), out=v)

    if fp_log1p:
        v = np.log1p(v).astype(np.float32)

    return v.astype(np.float32)


def smiles_to_fp(
    smi: str,
    radii,
    nbits_list,
    use_chirality=False,
    use_features=False,
    fp_log1p=True,
    clip_counts=None,
    cache=None,
):
    smi = str(smi).strip() if smi is not None else ""
    out_dim = int(sum(nbits_list))

    if smi == "" or smi.lower() == "nan":
        return np.zeros((out_dim,), dtype=np.float32)

    if cache is not None and smi in cache:
        return cache[smi]

    mol = _mol_from_smiles(smi)
    fp = mol_morgan_count_fp(
        mol,
        radii=radii,
        nbits_list=nbits_list,
        use_chirality=use_chirality,
        use_features=use_features,
        fp_log1p=fp_log1p,
        clip_counts=clip_counts,
    )

    if cache is not None:
        cache[smi] = fp
    return fp


# Feature builders

def rxn_to_feature_vector(
    rxn: str,
    gens,
    cache_key_base,
    cache,
    use_chirality=False,
    use_features=False,
    fp_log1p=True,
    clip_counts=None,
):
    """
    Ablation / 6-block reaction features:
      [react_sum, mid_sum, prod_sum, prod-react, prod-(react+mid), abs(...), 6 scalars]
    """
    rxn = normalize_rxn_text(rxn)
    left, mid, right = split_rxn_3parts(rxn)

    L = tokens_from_side(left)
    M = tokens_from_side(mid)
    P = tokens_from_side(right)

    D = sum(nb for _r, nb, _g in gens)
    r_vec = np.zeros((D,), dtype=np.float32)
    m_vec = np.zeros((D,), dtype=np.float32)
    p_vec = np.zeros((D,), dtype=np.float32)

    for smi in L:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            r_vec += mol_morgan_count_fp(
                mol,
                gens=gens,
                cache_key_base=cache_key_base,
                cache=cache,
                use_chirality=use_chirality,
                use_features=use_features,
                fp_log1p=fp_log1p,
                clip_counts=clip_counts,
            )

    for smi in M:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            m_vec += mol_morgan_count_fp(
                mol,
                gens=gens,
                cache_key_base=cache_key_base,
                cache=cache,
                use_chirality=use_chirality,
                use_features=use_features,
                fp_log1p=fp_log1p,
                clip_counts=clip_counts,
            )

    for smi in P:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            p_vec += mol_morgan_count_fp(
                mol,
                gens=gens,
                cache_key_base=cache_key_base,
                cache=cache,
                use_chirality=use_chirality,
                use_features=use_features,
                fp_log1p=fp_log1p,
                clip_counts=clip_counts,
            )

    b1 = r_vec
    b2 = m_vec
    b3 = p_vec
    b4 = (p_vec - r_vec)
    b5 = (p_vec - (r_vec + m_vec))
    b6 = np.abs(b5)

    scalars = np.array([
        float(len(L)),
        float(len(M)),
        float(len(P)),
        1.0 if len(L) > 0 else 0.0,
        1.0 if len(M) > 0 else 0.0,
        float(len(L) + len(M) + len(P)),
    ], dtype=np.float32)

    return np.concatenate([b1, b2, b3, b4, b5, b6, scalars], axis=0).astype(np.float32)

def rxn_to_feature(
    rxn: str,
    radii,
    nbits_list,
    use_chirality=False,
    use_features=False,
    fp_log1p=True,
    clip_counts=None,
    cache=None,
):
    """
      [prec_sum, prod_fp, diff, abs(diff), 4 scalars]
    """
    L, M, prod = rxn_to_roles(rxn)

    prec = L + M
    fp_dim = int(sum(nbits_list))
    prec_sum = np.zeros((fp_dim,), dtype=np.float32)

    for t in prec:
        if t and t.lower() != "nan":
            prec_sum += smiles_to_fp(
                t,
                radii=radii,
                nbits_list=nbits_list,
                use_chirality=use_chirality,
                use_features=use_features,
                fp_log1p=fp_log1p,
                clip_counts=clip_counts,
                cache=cache,
            )

    if prod:
        prod_fp = smiles_to_fp(
            prod,
            radii=radii,
            nbits_list=nbits_list,
            use_chirality=use_chirality,
            use_features=use_features,
            fp_log1p=fp_log1p,
            clip_counts=clip_counts,
            cache=cache,
        )
    else:
        prod_fp = np.zeros_like(prec_sum)

    diff = (prod_fp - prec_sum).astype(np.float32)
    adiff = np.abs(diff).astype(np.float32)

    nl = len(L)
    nm = len(M)
    nr = 1 if prod else 0
    nprec = len([t for t in prec if t])

    scalars = np.array([
        nl / 20.0,
        nm / 20.0,
        nr / 10.0,
        nprec / 20.0,
    ], dtype=np.float32)

    feat = np.concatenate([prec_sum, prod_fp, diff, adiff, scalars], axis=0).astype(np.float32)
    return feat


def build_feature_matrix_from_rxn_to_feature(
    rxns,
    radii,
    nbits_list,
    use_chirality=False,
    use_features=False,
    fp_log1p=True,
    clip_counts=None,
    cache=None,
):
    xdim = int(4 * sum(nbits_list) + 4)
    X = np.zeros((len(rxns), xdim), dtype=np.float32)
    for i, r in enumerate(rxns):
        X[i] = rxn_to_feature(
            r,
            radii=radii,
            nbits_list=nbits_list,
            use_chirality=use_chirality,
            use_features=use_features,
            fp_log1p=fp_log1p,
            clip_counts=clip_counts,
            cache=cache,
        )
    return X
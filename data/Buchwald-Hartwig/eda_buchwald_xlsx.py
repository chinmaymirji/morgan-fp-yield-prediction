#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Buchwald–Hartwig (Dreher–Doyle) XLSX EDA
Drop this file in the SAME folder as Dreher_and_Doyle_input_data.xlsx and run:

    python eda_buchwald_xlsx.py

No CLI args. Edit the CONFIG block only.
"""

# =========================
# CONFIG (EDIT THIS)
# =========================
XLSX_FILE   = "Dreher_and_Doyle_input_data.xlsx"
SHEET_NAME  = "FullCV_01"          # e.g., FullCV_01
OUTDIR      = "eda_buchwald_out"   # outputs written here
TOP_K_CATS  = 12                   # for ligand/base/additive/category plots
BINS_YIELD  = 50
BINS_LEN    = 40
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

# --------- helpers ----------
def _safe_mol(smiles):
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def rdkit_desc(mol):
    """Small, robust descriptor set (all numeric)."""
    if mol is None:
        return {
            "MolWt": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotB": np.nan,
            "Rings": np.nan,
            "AromRings": np.nan,
            "HeavyAtoms": np.nan,
        }
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotB": float(Lipinski.NumRotatableBonds(mol)),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "AromRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
    }

def corr_heatmap(df_num, title, outpath, max_cols=30):
    """
    - numeric only
    - drop all-NaN + constant columns
    - if too many columns, keep top |corr(y)| columns (+y)
    """
    df = df_num.select_dtypes(include=[np.number]).copy()
    df = df.dropna(axis=1, how="all")

    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const_cols, errors="ignore")

    if "y" in df.columns and len(df.columns) > max_cols:
        # keep the most related to y (by absolute Pearson corr)
        c = df.corr()["y"].drop("y", errors="ignore").abs().sort_values(ascending=False)
        keep = ["y"] + c.head(max_cols - 1).index.tolist()
        df = df[keep]

    corr = df.corr()
    cols = corr.columns.tolist()

    fig = plt.figure(figsize=(0.55 * len(cols) + 4, 0.55 * len(cols) + 4))
    ax = plt.gca()
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)
    return const_cols, cols

def bar_topk_mean_y(df, col, outpath, topk=12):
    vc = df[col].value_counts().head(topk)
    sub = df[df[col].isin(vc.index)].copy()
    means = sub.groupby(col)["y"].mean().loc[vc.index]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(means)), means.values)
    plt.xticks(range(len(means)), means.index, rotation=70, ha="right")
    plt.ylabel("Mean yield")
    plt.title(f"Mean yield by {col} (top {topk} by frequency)")
    plt.subplots_adjust(bottom=0.35)  # fixes tight_layout warning
    plt.savefig(outpath, dpi=220)
    plt.close()


def box_topk(df, col, outpath, topk=12):
    vc = df[col].value_counts().head(topk)
    sub = df[df[col].isin(vc.index)].copy()

    # avoid ragged numpy warning
    data = [sub[sub[col] == k]["y"].astype(float).tolist() for k in vc.index]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=vc.index, showfliers=False)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Yield")
    plt.title(f"Yield distribution by {col} (top {topk} by frequency)")
    plt.subplots_adjust(bottom=0.35)  # fixes tight_layout warning
    plt.savefig(outpath, dpi=220)
    plt.close()


def main():
    if not os.path.exists(XLSX_FILE):
        raise FileNotFoundError(f"Can't find {XLSX_FILE} in current folder: {os.getcwd()}")

    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_excel(XLSX_FILE, sheet_name=SHEET_NAME)

    needed = ["Ligand", "Additive", "Base", "Aryl halide", "Output"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    df = df[needed].copy()
    df = df.rename(columns={"Output": "y"})
    df["y"] = df["y"].astype(float)

    # basic summary
    y = df["y"].values
    summary = {
        "dataset": "Buchwald–Hartwig (Dreher–Doyle)",
        "xlsx": XLSX_FILE,
        "sheet": SHEET_NAME,
        "n_rows": int(len(df)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "frac_zero": float(np.mean(y == 0.0)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, "dataset_summary.csv"), index=False)

    # string-length features (simple)
    for c in ["Ligand", "Additive", "Base", "Aryl halide"]:
        df[c] = df[c].astype(str)
        df[f"{c}_smi_len"] = df[c].map(lambda s: len(str(s).strip()) if str(s).lower() != "nan" else np.nan)

    df["rxn_len"] = df["Aryl halide_smi_len"] + df["Ligand_smi_len"] + df["Additive_smi_len"] + df["Base_smi_len"]

    # RDKit descriptors per component
    desc_rows = []
    parse_ok = {c: 0 for c in ["Ligand", "Additive", "Base", "Aryl halide"]}

    for _, row in df.iterrows():
        out = {}
        for comp in ["Aryl halide", "Ligand", "Additive", "Base"]:
            mol = _safe_mol(row[comp])
            if mol is not None:
                parse_ok[comp] += 1
            d = rdkit_desc(mol)
            for k, v in d.items():
                out[f"{comp}_{k}"] = v
        desc_rows.append(out)

    desc_df = pd.DataFrame(desc_rows)
    full = pd.concat([df[["y", "rxn_len"] + [c for c in df.columns if c.endswith("_smi_len")]], desc_df], axis=1)
    full.to_csv(os.path.join(OUTDIR, "rdkit_features.csv"), index=False)

    # parse rates
    pr = {f"parse_rate_{k}": float(parse_ok[k] / max(1, len(df))) for k in parse_ok}
    pd.DataFrame([pr]).to_csv(os.path.join(OUTDIR, "rdkit_parse_rates.csv"), index=False)

    # --- plots ---
    # yield hist
    plt.figure()
    plt.hist(y, bins=BINS_YIELD)
    plt.title("Yield distribution")
    plt.xlabel("Yield")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "yield_hist.png"), dpi=220)
    plt.close()

    # yield CDF
    ys = np.sort(y[np.isfinite(y)])
    plt.figure()
    plt.plot(ys, np.linspace(0, 1, len(ys), endpoint=True))
    plt.title("Yield CDF")
    plt.xlabel("Yield")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "yield_cdf.png"), dpi=220)
    plt.close()

    # rxn length distribution
    plt.figure()
    plt.hist(df["rxn_len"].dropna().values, bins=BINS_LEN)
    plt.title("Reaction length distribution (sum of component SMILES lengths)")
    plt.xlabel("Reaction length (chars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "rxn_len_hist.png"), dpi=220)
    plt.close()

    # yield vs rxn length
    plt.figure()
    plt.scatter(df["rxn_len"].values, y, s=10, alpha=0.5)
    plt.title("Yield vs reaction length")
    plt.xlabel("Reaction length (chars)")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "y_vs_rxn_len.png"), dpi=220)
    plt.close()

    # correlation heatmaps
    const1, _ = corr_heatmap(
        full[["y", "rxn_len"] + [c for c in full.columns if c.endswith("_smi_len")]],
        "Correlation: yield vs SMILES length features",
        os.path.join(OUTDIR, "corr_smiles_len.png"),
        max_cols=25
    )
    const2, _ = corr_heatmap(
        pd.concat([df[["y"]], desc_df], axis=1),
        "Correlation: yield vs RDKit descriptors (top correlated subset)",
        os.path.join(OUTDIR, "corr_rdkit_desc.png"),
        max_cols=30
    )

    # categorical plots (top-K)
    for col in ["Ligand", "Base", "Additive", "Aryl halide"]:
        bar_topk_mean_y(df, col, os.path.join(OUTDIR, f"mean_y_by_{col.replace(' ','_')}.png"), topk=TOP_K_CATS)
        box_topk(df, col, os.path.join(OUTDIR, f"box_y_by_{col.replace(' ','_')}.png"), topk=TOP_K_CATS)

    print("[DONE] Wrote outputs to:", OUTDIR)
    print("[INFO] RDKit parse rates written to rdkit_parse_rates.csv")

if __name__ == "__main__":
    main()


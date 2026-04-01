#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drop this script into the SAME folder as Dreher_and_Doyle_input_data.xlsx
and run:

  python eda_buchwald_xlsx_local.py

Outputs go to ./eda_out/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT IF NEEDED)
# =========================
XLSX_NAME = ""          # leave blank to auto-pick first .xlsx in folder
PREFERRED_SHEETS = ["FullCV_01", "FullCV_02", "FullCV_03", "FullCV"]
OUTDIR = "eda_out"
DATASET_NAME = "Buchwald–Hartwig (Dreher–Doyle)"
DEDUP = "none"          # "none", "mean", "median" (dedup on built rxn string)
# =========================


def find_xlsx_here():
    if XLSX_NAME and os.path.exists(XLSX_NAME):
        return XLSX_NAME
    xlsxs = [f for f in os.listdir(".") if f.lower().endswith((".xlsx", ".xls"))]
    if not xlsxs:
        raise FileNotFoundError("No .xlsx/.xls found in current folder.")
    xlsxs.sort()
    return xlsxs[0]


def pick_sheet(path):
    xf = pd.ExcelFile(path)
    for s in PREFERRED_SHEETS:
        if s in xf.sheet_names:
            return s
    return xf.sheet_names[0]


def normalize_rxn_text(s: str) -> str:
    s = str(s).replace(" ", "").replace("|", "").replace("~", ".")
    parts = s.split(">")
    if len(parts) >= 3:
        s = ">".join([parts[0], parts[1], ">".join(parts[2:])])
    elif len(parts) == 2:
        s = parts[0] + ">" + ">" + parts[1]
    return s


def build_buchwald_rxn(row) -> str:
    """
    Build a 3-part reaction string:
      reactants > reagents > products

    Here we use:
      reactant: Aryl halide
      reagents: Ligand, Additive, Base
      products: blank (not provided in this table)
    """
    aryl = str(row.get("Aryl halide", "")).strip()
    lig  = str(row.get("Ligand", "")).strip()
    add  = str(row.get("Additive", "")).strip()
    base = str(row.get("Base", "")).strip()

    reagents = ".".join([x for x in [lig, add, base] if x and x.lower() != "nan"])
    rxn = f"{aryl}>{reagents}>"
    return normalize_rxn_text(rxn)


def split_rxn_3(rxn: str):
    parts = str(rxn).split(">")
    if len(parts) < 3:
        parts = (parts + ["", "", ""])[:3]
    else:
        parts = [parts[0], parts[1], ">".join(parts[2:])]
    return parts[0], parts[1], parts[2]


def derived_features(rxn: str):
    r, m, p = split_rxn_3(rxn)
    r_list = [x for x in r.split(".") if x]
    m_list = [x for x in m.split(".") if x]
    p_list = [x for x in p.split(".") if x]

    tokens = r_list + m_list + p_list
    uniq = set(tokens)
    token_lens = [len(t) for t in tokens] if tokens else [0]

    return {
        "char_len": len(rxn),
        "n_tokens_total": len(tokens),
        "n_reactants": len(r_list),
        "n_reagents": len(m_list),
        "n_products": len(p_list),
        "uniq_tokens": len(uniq),
        "mean_token_len": float(np.mean(token_lens)),
        "max_token_len": int(np.max(token_lens)),
    }


def safe_hist(x, title, xlabel, outpath, bins=50):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    plt.figure()
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def safe_scatter(x, y, title, xlabel, ylabel, outpath):
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    plt.figure()
    plt.scatter(x[ok], y[ok], s=10, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def corr_heatmap(num_df: pd.DataFrame, title: str, outpath: str):
    df = num_df.select_dtypes(include=[np.number]).copy()
    df = df.dropna(axis=1, how="all")
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const_cols, errors="ignore")

    corr = df.corr()
    cols = corr.columns.tolist()

    fig = plt.figure(figsize=(0.7 * len(cols) + 3, 0.7 * len(cols) + 3))
    ax = plt.gca()
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)

    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

    return const_cols


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    xlsx = find_xlsx_here()
    sheet = pick_sheet(xlsx)

    print("[INFO] XLSX:", xlsx)
    print("[INFO] Sheet:", sheet)

    df = pd.read_excel(xlsx, sheet_name=sheet)

    needed = {"Ligand", "Additive", "Base", "Aryl halide", "Output"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise RuntimeError(
            "This doesn't look like Dreher–Doyle input schema.\n"
            f"Missing columns: {missing}\n"
            f"Found columns: {list(df.columns)[:30]} ..."
        )

    # Build rxn + y
    df2 = df[list(needed)].copy()
    df2["rxn"] = df2.apply(build_buchwald_rxn, axis=1)
    df2["y"] = pd.to_numeric(df2["Output"], errors="coerce")

    # Drop NaN targets
    before = len(df2)
    df2 = df2[df2["y"].notna()].copy()
    after = len(df2)
    if after != before:
        print(f"[INFO] Dropped rows with NaN Output: {before} -> {after}")

    df2 = df2[["rxn", "y"]]

    # Optional dedup
    if DEDUP != "none":
        agg = np.mean if DEDUP == "mean" else np.median
        b = len(df2)
        df2 = df2.groupby("rxn", as_index=False)["y"].agg(agg)
        print(f"[INFO] Dedup={DEDUP}: {b} -> {len(df2)}")

    y = df2["y"].astype(float).values

    # Summary
    summary = {
        "dataset": DATASET_NAME,
        "source_file": os.path.basename(xlsx),
        "sheet": sheet,
        "n_rows": int(len(df2)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "frac_zero": float(np.mean(y == 0.0)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, "dataset_summary.csv"), index=False)
    print("Wrote:", os.path.join(OUTDIR, "dataset_summary.csv"))

    # Derived features
    feat_df = pd.DataFrame([derived_features(r) for r in df2["rxn"].tolist()])
    feat_df.to_csv(os.path.join(OUTDIR, "derived_features.csv"), index=False)
    print("Wrote:", os.path.join(OUTDIR, "derived_features.csv"))

    # Correlation heatmap (fixed NaNs)
    num_df = pd.concat([df2[["y"]], feat_df], axis=1)
    dropped = corr_heatmap(
        num_df,
        title=f"{DATASET_NAME}: correlation (yield vs derived rxn-text features)",
        outpath=os.path.join(OUTDIR, "corr_heatmap.png"),
    )
    print("Wrote:", os.path.join(OUTDIR, "corr_heatmap.png"))
    if dropped:
        print("[INFO] Dropped constant columns (avoids white NaN blocks):", dropped)

    # Plots
    safe_hist(y, f"{DATASET_NAME}: yield distribution", "Yield",
              os.path.join(OUTDIR, "yield_hist.png"), bins=50)

    # Yield CDF
    ys = np.sort(y[np.isfinite(y)])
    plt.figure()
    plt.plot(ys, np.linspace(0, 1, len(ys), endpoint=True))
    plt.title(f"{DATASET_NAME}: yield CDF")
    plt.xlabel("Yield")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "yield_cdf.png"), dpi=200)
    plt.close()
    print("Wrote:", os.path.join(OUTDIR, "yield_cdf.png"))

    safe_scatter(feat_df["char_len"].values, y,
                 f"{DATASET_NAME}: yield vs reaction length",
                 "Reaction string length (chars)", "Yield",
                 os.path.join(OUTDIR, "y_vs_charlen.png"))

    safe_scatter(feat_df["n_tokens_total"].values, y,
                 f"{DATASET_NAME}: yield vs #components",
                 "Total components (tokens)", "Yield",
                 os.path.join(OUTDIR, "y_vs_num_tokens.png"))

    safe_scatter(feat_df["uniq_tokens"].values, y,
                 f"{DATASET_NAME}: yield vs unique components",
                 "#Unique components", "Yield",
                 os.path.join(OUTDIR, "y_vs_unique_tokens.png"))

    safe_hist(feat_df["char_len"].values,
              f"{DATASET_NAME}: reaction length distribution",
              "reaction length (chars)",
              os.path.join(OUTDIR, "hist_char_len.png"),
              bins=40)

    safe_hist(feat_df["n_tokens_total"].values,
              f"{DATASET_NAME}: #components distribution",
              "#components (tokens)",
              os.path.join(OUTDIR, "hist_num_tokens.png"),
              bins=40)

    print("\n[DONE] All outputs in:", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()


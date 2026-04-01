#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import csv
import numpy as np
import pandas as pd

XLSX_NAME = "Dreher_and_Doyle_input_data.xlsx"
OUT_DIR   = "random_splits"
N_SPLITS  = 10
SEED_BASE = 123
PREFER_FULLCV_SHEETS = True

TIMING_CSV = "split_generation_timing.csv"


def normalize_rxn_text(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "")
    s = s.replace("|", "")
    return s


def build_rxn_row(lig, add, base, aryl) -> str:
    left = ".".join([str(lig), str(add), str(base), str(aryl)])
    rxn = f"{left}>>"
    return normalize_rxn_text(rxn)


def output_to_fraction(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    ymax = float(np.nanmax(y))
    if ymax > 1.5 and ymax <= 100.0:
        y = y / 100.0
    return y


def find_fullcv_sheets(sheet_names):
    pat = re.compile(r"^FullCV[_\s]*0?(\d+)$", re.IGNORECASE)
    hits = []
    for s in sheet_names:
        m = pat.match(s.strip())
        if m:
            hits.append((int(m.group(1)), s))
    hits.sort(key=lambda x: x[0])
    return [name for _, name in hits]


def load_sheet_as_rxn_y(xlsx_path, sheet_name):
    t0 = time.perf_counter()

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    needed = ["Ligand", "Additive", "Base", "Aryl halide", "Output"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sheet '{sheet_name}' missing columns: {missing}. Found: {list(df.columns)}")

    rxn = [
        build_rxn_row(l, a, b, ar)
        for l, a, b, ar in zip(df["Ligand"], df["Additive"], df["Base"], df["Aryl halide"])
    ]
    y = output_to_fraction(df["Output"].to_numpy())

    out = pd.DataFrame({"rxn": rxn, "y": y})
    out = out.dropna(subset=["rxn", "y"]).reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    return out, elapsed


def build_full_dataset_from_plates(xlsx_path):
    t0 = time.perf_counter()

    sheets = pd.ExcelFile(xlsx_path).sheet_names
    if "Plates1-3" in sheets and "Plate4" in sheets:
        df_a, t_a = load_sheet_as_rxn_y(xlsx_path, "Plates1-3")
        df_b, t_b = load_sheet_as_rxn_y(xlsx_path, "Plate4")
        df = pd.concat([df_a, df_b], axis=0, ignore_index=True)
        elapsed = time.perf_counter() - t0
        return df, elapsed, {"Plates1-3_time_s": t_a, "Plate4_time_s": t_b}

    for s in sheets:
        try:
            df, t_s = load_sheet_as_rxn_y(xlsx_path, s)
            if len(df) > 0:
                elapsed = time.perf_counter() - t0
                return df, elapsed, {f"{s}_time_s": t_s}
        except Exception:
            continue

    raise RuntimeError("Could not find any sheet with columns: Ligand, Additive, Base, Aryl halide, Output")


def main():
    total_start = time.perf_counter()

    here = os.getcwd()
    xlsx_path = os.path.join(here, XLSX_NAME)
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"Could not find {XLSX_NAME} in: {here}")

    os.makedirs(OUT_DIR, exist_ok=True)

    t_xls = time.perf_counter()
    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names
    xls_load_time = time.perf_counter() - t_xls

    fullcv_sheets = find_fullcv_sheets(sheet_names)
    use_fullcv = PREFER_FULLCV_SHEETS and len(fullcv_sheets) >= N_SPLITS

    print("[INFO] cwd:", here)
    print("[INFO] xlsx:", xlsx_path)
    print("[INFO] sheets:", len(sheet_names))
    print("[INFO] using FullCV sheets:", use_fullcv)

    split_rows = []
    total_rows_written = 0

    if use_fullcv:
        for i in range(N_SPLITS):
            sh = fullcv_sheets[i]

            t_split = time.perf_counter()
            df, sheet_time = load_sheet_as_rxn_y(xlsx_path, sh)
            out_path = os.path.join(OUT_DIR, f"random_split_{i}.tsv")
            df.to_csv(out_path, sep="\t", index=False)
            split_elapsed = time.perf_counter() - t_split

            nrows = len(df)
            total_rows_written += nrows

            split_rows.append({
                "split_idx": i,
                "sheet": sh,
                "rows": nrows,
                "sheet_parse_time_s": sheet_time,
                "total_split_time_s": split_elapsed,
                "time_per_row_ms": 1000.0 * split_elapsed / max(nrows, 1),
            })

            print(f"[OK] wrote {out_path} (rows={nrows}) from sheet={sh} | {split_elapsed:.3f}s")

    else:
        base, base_build_time, sub_times = build_full_dataset_from_plates(xlsx_path)
        n = len(base)
        print("[INFO] base rows:", n)
        print(f"[INFO] base build time: {base_build_time:.3f}s")

        for i in range(N_SPLITS):
            t_split = time.perf_counter()

            rng = np.random.RandomState(SEED_BASE + i)
            perm = rng.permutation(n)
            df = base.iloc[perm].reset_index(drop=True)

            out_path = os.path.join(OUT_DIR, f"random_split_{i}.tsv")
            df.to_csv(out_path, sep="\t", index=False)

            split_elapsed = time.perf_counter() - t_split
            nrows = len(df)
            total_rows_written += nrows

            split_rows.append({
                "split_idx": i,
                "sheet": f"seed_{SEED_BASE+i}",
                "rows": nrows,
                "sheet_parse_time_s": base_build_time if i == 0 else 0.0,
                "total_split_time_s": split_elapsed,
                "time_per_row_ms": 1000.0 * split_elapsed / max(nrows, 1),
            })

            print(f"[OK] wrote {out_path} (rows={nrows}) seed={SEED_BASE+i} | {split_elapsed:.3f}s")

    total_elapsed = time.perf_counter() - total_start
    avg_split_time = np.mean([r["total_split_time_s"] for r in split_rows])
    avg_row_ms = 1000.0 * total_elapsed / max(total_rows_written, 1)

    timing_csv_path = os.path.join(here, TIMING_CSV)
    with open(timing_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split_idx", "sheet", "rows", "sheet_parse_time_s", "total_split_time_s", "time_per_row_ms"]
        )
        writer.writeheader()
        writer.writerows(split_rows)

    print("\n=== TIMING SUMMARY ===")
    print(f"Excel workbook load time: {xls_load_time:.3f} s")
    print(f"Total split generation time: {total_elapsed:.3f} s")
    print(f"Average time per split: {avg_split_time:.3f} s")
    print(f"Total rows written across all splits: {total_rows_written}")
    print(f"Average time per row: {avg_row_ms:.4f} ms")
    print(f"Timing CSV saved to: {timing_csv_path}")
    print("[DONE] Splits created in:", os.path.join(here, OUT_DIR))


if __name__ == "__main__":
    main()
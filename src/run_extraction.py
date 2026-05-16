# -*- coding: utf-8 -*-
"""
src/run_extraction.py  --  Batch feature extraction for all samples.

Usage (run from project root):
    python src/run_extraction.py                   # whole-genome, sequential, 50/batch
    python src/run_extraction.py --chrom chr1      # fast dev run
    python src/run_extraction.py --workers 4       # parallel (Linux/Mac only; Windows uses sequential)
    python src/run_extraction.py --no-prompt       # unattended

Resume: re-run the same command -- completed files are skipped automatically.

Output:
    data/processed/features_checkpoint.csv    # grows as each file finishes
    data/processed/features_final.csv         # written when all batches complete
"""

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path so config and src.* imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import config
from src.load_sample import load_and_filter
from src.features import extract_features

CHECKPOINT = config.DATA_PROC / "features_checkpoint.csv"
FINAL_OUT  = config.DATA_PROC / "features_final.csv"


# ── Worker (module-level for multiprocessing) ─────────────────────
def _worker(args: tuple) -> dict:
    path, label, chrom = args
    try:
        df    = load_and_filter(path, chrom=chrom)
        feats = extract_features(df, label)
        feats["filename"] = Path(path).name
        feats["status"]   = "ok"
    except Exception as exc:
        feats = {
            "filename": Path(path).name,
            "label":    label,
            "status":   f"ERROR: {exc}",
        }
    return feats


# ── Checkpoint helpers ────────────────────────────────────────────
def _done_filenames() -> set:
    if CHECKPOINT.exists():
        df = pd.read_csv(CHECKPOINT, usecols=["filename"])
        return set(df["filename"].tolist())
    return set()


def _append_checkpoint(row: dict) -> None:
    write_header = not CHECKPOINT.exists()
    pd.DataFrame([row]).to_csv(CHECKPOINT, mode="a", header=write_header, index=False)


def _get_label(path: Path) -> str:
    for part in path.parts:
        if part.lower() in ("healthy", "cancer"):
            return part.lower()
    return "unknown"


# ── Batch runner ──────────────────────────────────────────────────
def _print_result(result, done, total, t0):
    elapsed   = time.time() - t0
    remaining = (total - done) * (elapsed / done) if done else 0
    slr = result.get("short_long_ratio", "n/a")
    slr = f"{slr:.4f}" if isinstance(slr, float) else slr
    print(
        f"  [{done:>3}/{total}]  {result['filename']:<42}"
        f"  S/L={slr}  {result['status']}"
        f"  ({remaining:.0f}s left)",
        flush=True,
    )


def run_batch(files, chrom, n_workers, batch_num, n_batches):
    print(f"\n{'='*60}", flush=True)
    print(f"  BATCH {batch_num}/{n_batches}  --  {len(files)} files  --  {n_workers} workers", flush=True)
    print(f"{'='*60}", flush=True)

    results = []
    done    = 0
    t0      = time.time()

    use_parallel = n_workers > 1 and os.name != "nt"

    if use_parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_worker, (p, _get_label(p), chrom)): p
                for p in files
            }
            for future in as_completed(futures):
                result = future.result()
                _append_checkpoint(result)
                results.append(result)
                done += 1
                _print_result(result, done, len(files), t0)
    else:
        for p in files:
            result = _worker((p, _get_label(p), chrom))
            _append_checkpoint(result)
            results.append(result)
            done += 1
            _print_result(result, done, len(files), t0)

    return results


def _batch_summary(results):
    ok     = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if "ERROR" in r.get("status", "")]
    all_done = len(pd.read_csv(CHECKPOINT)) if CHECKPOINT.exists() else 0
    total    = sum(1 for _ in config.DATA_RAW.rglob("*.frag.tsv.bgz"))
    print(f"\n  Done: {len(ok)} ok   {len(errors)} errors   Overall: {all_done}/{total}")
    for r in errors:
        print(f"  ERROR: {r['filename']}: {r['status']}")


def _write_final():
    if not CHECKPOINT.exists():
        return
    df = pd.read_csv(CHECKPOINT)
    # Keep last (most recent) entry per file; drop error rows
    df = df.drop_duplicates(subset=["filename"], keep="last")
    clean = df[~df["status"].str.startswith("ERROR", na=False)]
    # Overwrite checkpoint with deduplicated version
    clean.to_csv(CHECKPOINT, index=False)
    clean.to_csv(FINAL_OUT, index=False)
    print(f"\nFinal feature matrix: {len(clean)} samples  -->  {FINAL_OUT}", flush=True)


# ── Main ──────────────────────────────────────────────────────────
def main(chrom, n_workers, batch_size, no_prompt):
    all_files = sorted(config.DATA_RAW.rglob("*.frag.tsv.bgz"))
    if not all_files:
        print(f"No .frag.tsv.bgz files found under {config.DATA_RAW}")
        return

    done = _done_filenames()
    todo = [p for p in all_files if p.name not in done]
    label_counts = {}
    for p in all_files:
        lbl = _get_label(p)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\ncfDNA Feature Extraction  --  Phase A")
    print(f"  Total files   : {len(all_files)}  {label_counts}")
    print(f"  Already done  : {len(done)}")
    print(f"  To process    : {len(todo)}")
    print(f"  Batch size    : {batch_size}   Workers: {n_workers}")
    print(f"  Chrom filter  : {chrom or 'ALL (whole-genome)'}")
    print(f"  Output        : {FINAL_OUT}")

    if not todo:
        print("\nAll files already processed.")
        _write_final()
        return

    batches   = [todo[i:i+batch_size] for i in range(0, len(todo), batch_size)]
    n_batches = len(batches)

    for i, batch in enumerate(batches, 1):
        results = run_batch(batch, chrom, n_workers, i, n_batches)
        _batch_summary(results)

        if i == n_batches:
            print("\nAll batches complete.")
            _write_final()
            break

        if no_prompt:
            print(f"\nContinuing to batch {i+1} ...")
        else:
            ans = input(f"\nContinue to batch {i+1}/{n_batches}? [y / n / skip-prompts]: ").strip().lower()
            if ans == "n":
                _write_final()
                break
            if ans == "skip-prompts":
                no_prompt = True


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--chrom",      default=None)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--no-prompt",  action="store_true")
    args = p.parse_args()
    main(args.chrom, args.workers, args.batch_size, args.no_prompt)

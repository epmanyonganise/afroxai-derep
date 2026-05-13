#!/usr/bin/env python
"""
build_sea_index.py — One-time script to build the local SEA target-ligand index.

Run this ONCE before using the SEA feature in the app:

    # Standard build (~25 min, pChEMBL >= 5, broad coverage)
    python build_sea_index.py

    # Fast build (~10 min, pChEMBL >= 6, high-confidence actives only)
    python build_sea_index.py --fast

    # Custom options
    python build_sea_index.py --pchembl 6.0 --max-ligands 150 --output artifacts/sea_index.pkl.gz

The index is saved to artifacts/sea_index.pkl.gz and loaded automatically
by the Streamlit app on subsequent runs.
"""
import argparse
import sys
import time
from pathlib import Path

from sea_local import build_sea_index

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Build SEA ChEMBL target-ligand index",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--pchembl", type=float, default=5.0,
    help="Minimum pChEMBL cutoff (5 ≈ IC50 ≤ 10 µM; 6 ≈ ≤ 1 µM)",
)
parser.add_argument(
    "--min-ligands", type=int, default=10,
    help="Discard targets with fewer known actives than this",
)
parser.add_argument(
    "--max-ligands", type=int, default=300,
    help="Maximum ligands per target (caps index size)",
)
parser.add_argument(
    "--organism", default="Homo sapiens",
    help="ChEMBL organism filter",
)
parser.add_argument(
    "--output", default="artifacts/sea_index.pkl.gz",
    help="Output path for the compressed index",
)
parser.add_argument(
    "--fast", action="store_true",
    help="Quick build: forces pChEMBL>=6 and max-ligands=150",
)
args = parser.parse_args()

if args.fast:
    args.pchembl     = max(args.pchembl, 6.0)
    args.max_ligands = min(args.max_ligands, 150)

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("AfroXai-Derep  —  SEA Index Builder")
print("=" * 60)
print(f"  organism    : {args.organism}")
print(f"  pChEMBL ≥   : {args.pchembl}")
print(f"  min ligands : {args.min_ligands}")
print(f"  max ligands : {args.max_ligands}")
print(f"  output      : {output}")
print()

t0 = time.time()

payload = build_sea_index(
    cache_path     = output,
    organism       = args.organism,
    pchembl_cutoff = args.pchembl,
    min_ligands    = args.min_ligands,
    max_ligands    = args.max_ligands,
    on_progress    = print,
)

elapsed = time.time() - t0
n_targets = len(payload["index"])
bg = payload["bg"]

print()
print("=" * 60)
print(f"Done in {elapsed / 60:.1f} min")
print(f"  Targets indexed : {n_targets}")
print(f"  mu_slope        : {bg['mu_slope']:.5f}")
print(f"  beta_coef       : {bg['beta_coef']:.5f}")
print(f"  tc_cutoff       : {bg['tc_cutoff']}")
print(f"  Saved to        : {output}")
print()
print("You can now restart the Streamlit app — SEA predictions will be")
print("available automatically on every Analyse click.")

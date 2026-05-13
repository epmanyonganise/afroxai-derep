"""
sea_local.py — Local Similarity Ensemble Approach (SEA) for target prediction.

Reference: Keiser et al. (2007) Nature 439, doi:10.1038/nature05507

Algorithm
---------
For each target T with N known ligands {L_i}:
  1.  tc_i  = Tanimoto(query, L_i)  [ECFP4, 2048-bit]
  2.  RS    = Σ tc_i  where tc_i >= TC_CUTOFF
  3.  Null model: RS ~ Gumbel(μ = mu_slope·N,  β = beta_coef·√N)
      Parameters are empirically fitted from random compound pairs in the
      index corpus at build time, so they reflect the actual chemical space.
  4.  p-value = 1 − GumbelCDF(RS; μ, β)
  5.  E-value = p × |targets|    (Bonferroni correction for multiple testing)

Targets are ranked by ascending E-value; lower means more significant.

Usage
-----
Build the index once (takes ~15-30 min, saved to artifacts/sea_index.pkl.gz):
    python build_sea_index.py

Then predict:
    from sea_local import load_sea_index, predict_sea
    payload = load_sea_index("artifacts/sea_index.pkl.gz")
    result  = predict_sea("CC(=O)Oc1ccccc1C(=O)O", payload)
"""
from __future__ import annotations

import gzip
import logging
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────
TC_CUTOFF     = 0.30   # Tanimoto threshold (ECFP4 standard; 0.3 balances sensitivity)
MIN_LIGANDS   = 10     # skip targets with fewer confirmed actives
MAX_LIGANDS   = 300    # cap per target (keeps index size and query time bounded)
DEFAULT_TOP_N = 20
_CHEMBL_BASE  = "https://www.ebi.ac.uk/chembl/api/data"


# ── Fingerprint helpers ───────────────────────────────────────────────────────

def _fp(smiles: str) -> np.ndarray | None:
    """ECFP4 (Morgan radius=2, 2048 bits) as uint8 array, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _tc_batch(q: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """
    Vectorised Tanimoto: q (2048,) vs refs (N, 2048) → float32 (N,).
    Uses the bit-count identity: Tc = |A∩B| / |A∪B|
    """
    qi    = q.astype(np.int32)
    ri    = refs.astype(np.int32)
    inter = ri @ qi
    union = qi.sum() + ri.sum(axis=1) - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ── ChEMBL API helper ─────────────────────────────────────────────────────────

def _chembl_get(endpoint: str, params: dict,
                retries: int = 3, timeout: int = 30) -> dict:
    url = f"{_CHEMBL_BASE}/{endpoint}.json"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params,
                             headers={"Accept": "application/json"},
                             timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ── Background model calibration ──────────────────────────────────────────────

def _calibrate(
    corpus_fps : np.ndarray,
    tc_cutoff  : float               = TC_CUTOFF,
    n_sample   : int                 = 400,
    set_sizes  : tuple[int, ...]     = (10, 20, 50, 100, 200),
    n_rounds   : int                 = 300,
    seed       : int                 = 42,
) -> dict:
    """
    Empirically fit Gumbel null-model parameters from random draws of the
    index corpus.  For each set size N we simulate *n_rounds* random
    (query, ligand-set-of-N) draws, compute the sum-of-Tc RS, fit a Gumbel
    distribution, then regress:
        mu(N)   = mu_slope  × N       (linear in N)
        beta(N) = beta_coef × √N      (linear in √N)

    If scipy is unavailable, hand-tuned fallback values are returned.
    """
    try:
        from scipy.stats import gumbel_r
    except ImportError:
        log.warning("scipy not found — using fallback Gumbel parameters.")
        return {"mu_slope": 0.08, "beta_coef": 0.035, "tc_cutoff": tc_cutoff}

    rng  = np.random.default_rng(seed)
    pick = rng.choice(len(corpus_fps), min(n_sample, len(corpus_fps)), replace=False)
    pool = corpus_fps[pick]

    Ns, mus, betas = [], [], []
    for N in set_sizes:
        if N + 1 > len(pool):
            continue
        rs_vals = []
        for _ in range(n_rounds):
            q_i   = rng.integers(0, len(pool))
            l_idx = rng.choice(len(pool), N, replace=False)
            tcs   = _tc_batch(pool[q_i], pool[l_idx])
            rs_vals.append(float(tcs[tcs >= tc_cutoff].sum()))
        loc, scale = gumbel_r.fit(rs_vals)
        Ns.append(N); mus.append(loc); betas.append(scale)

    Ns_arr    = np.array(Ns, dtype=float)
    mu_slope  = max(float(np.polyfit(Ns_arr,           mus,   1)[0]), 1e-8)
    beta_coef = max(float(np.polyfit(np.sqrt(Ns_arr),  betas, 1)[0]), 1e-8)

    log.info("Calibration complete: mu_slope=%.5f  beta_coef=%.5f",
             mu_slope, beta_coef)
    return {"mu_slope": mu_slope, "beta_coef": beta_coef, "tc_cutoff": tc_cutoff}


# ── Index build ───────────────────────────────────────────────────────────────

def build_sea_index(
    cache_path      : str | Path,
    organism        : str   = "Homo sapiens",
    pchembl_cutoff  : float = 5.0,
    min_ligands     : int   = MIN_LIGANDS,
    max_ligands     : int   = MAX_LIGANDS,
    on_progress     : Callable[[str], None] | None = None,
) -> dict:
    """
    Download binding activities from ChEMBL, compute ECFP4 fingerprints per
    target, calibrate the Gumbel null model, and write a compressed index.

    Only needs to run **once**.  Typical runtimes:
        pChEMBL ≥ 6  →  ~10 min  (recommended fast build)
        pChEMBL ≥ 5  →  ~25 min  (broader coverage)

    Parameters
    ----------
    cache_path     : where to write artifacts/sea_index.pkl.gz
    organism       : ChEMBL organism filter (default "Homo sapiens")
    pchembl_cutoff : minimum pChEMBL value (5 ≈ IC50 ≤ 10 µM; 6 ≈ ≤ 1 µM)
    min_ligands    : discard targets with fewer actives than this
    max_ligands    : cap ligands per target (sorted by best pChEMBL)
    on_progress    : optional callback(str) for progress messages

    Returns
    -------
    payload dict — same structure as load_sea_index()
    """
    cache_path = Path(cache_path)

    def _p(msg: str) -> None:
        log.info(msg)
        if on_progress:
            on_progress(msg)

    _p(f"Building SEA index  organism={organism!r}  "
       f"pChEMBL≥{pchembl_cutoff}  min={min_ligands}  max={max_ligands}")

    # ── 1. Fetch binding activities in pages ──────────────────────────────────
    activities: dict[str, dict] = {}   # tid → {name, target_type, smiles: set}
    offset, limit, page = 0, 500, 0

    while True:
        page += 1
        _p(f"  Fetching ChEMBL page {page} (offset {offset})…")
        try:
            data = _chembl_get("activity", {
                "target_organism":    organism,
                "pchembl_value__gte": pchembl_cutoff,
                "assay_type":         "B",
                "limit":              limit,
                "offset":             offset,
            })
        except Exception as exc:
            _p(f"  API error: {exc} — stopping fetch at page {page}.")
            break

        for act in data.get("activities", []):
            tid = act.get("target_chembl_id")
            smi = act.get("canonical_smiles")
            if not tid or not smi:
                continue
            if tid not in activities:
                activities[tid] = {
                    "name":        act.get("target_pref_name", tid),
                    "target_type": act.get("target_type", "SINGLE PROTEIN"),
                    "smiles":      set(),
                }
            activities[tid]["smiles"].add(smi)

        if data.get("page_meta", {}).get("next") is None:
            break
        offset += limit

    total_acts = sum(len(v["smiles"]) for v in activities.values())
    _p(f"Fetched {total_acts} activities across {len(activities)} targets.")

    # ── 2. Compute ECFP4 fingerprints ─────────────────────────────────────────
    index:  dict[str, dict]   = {}
    corpus: list[np.ndarray]  = []

    for i, (tid, meta) in enumerate(activities.items()):
        smiles_list = list(meta["smiles"])[:max_ligands]
        fps = [arr for smi in smiles_list
               if (arr := _fp(smi)) is not None]
        if len(fps) < min_ligands:
            continue
        fps_arr = np.array(fps, dtype=np.uint8)
        index[tid] = {
            "name":        meta["name"],
            "target_type": meta["target_type"],
            "organism":    organism,
            "fps":         fps_arr,
        }
        corpus.extend(fps[:50])             # ≤50 fps/target for calibration
        if i % 100 == 0:
            _p(f"  Fingerprints: {i}/{len(activities)} targets  "
               f"({len(index)} qualify)…")

    _p(f"Index: {len(index)} targets with ≥{min_ligands} active ligands.")

    # ── 3. Calibrate Gumbel null model ────────────────────────────────────────
    _p("Calibrating Gumbel background model (sampling random pairs)…")
    corpus_arr = np.array(corpus[:2000], dtype=np.uint8)
    bg = _calibrate(corpus_arr, tc_cutoff=TC_CUTOFF)
    _p(f"  mu_slope={bg['mu_slope']:.5f}  beta_coef={bg['beta_coef']:.5f}")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    payload = {"index": index, "bg": bg, "organism": organism}
    with gzip.open(cache_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    size_mb = cache_path.stat().st_size / 1e6
    _p(f"Saved → {cache_path}  ({size_mb:.1f} MB)")
    return payload


def load_sea_index(cache_path: str | Path) -> dict:
    """Load a previously built SEA index from disk."""
    with gzip.open(Path(cache_path), "rb") as fh:
        return pickle.load(fh)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_sea(
    query_smiles : str,
    payload      : dict,
    top_n        : int   = DEFAULT_TOP_N,
    e_cutoff     : float = 100.0,
) -> dict:
    """
    Run local SEA prediction against the prebuilt index.

    Parameters
    ----------
    query_smiles : SMILES string of the query molecule
    payload      : dict from load_sea_index() or build_sea_index()
    top_n        : return at most this many predictions
    e_cutoff     : include only targets with E-value < e_cutoff
                   (100 = very lenient; 1 = nominally significant; 0.05 = strict)

    Returns
    -------
    dict with keys:
        status              "OK" | "NO_PREDICTIONS" | "INVALID_SMILES" | "ERROR"
        predictions         list[dict], sorted by ascending e_value
        n_targets_searched  int
        query_smiles        str
    """
    try:
        from scipy.stats import gumbel_r
    except ImportError:
        return {
            "status": "ERROR", "predictions": [],
            "reason": "scipy required — run: pip install scipy",
        }

    query_fp = _fp(query_smiles)
    if query_fp is None:
        return {"status": "INVALID_SMILES", "predictions": []}

    index     = payload["index"]
    bg        = payload["bg"]
    mu_slope  = bg["mu_slope"]
    beta_coef = bg["beta_coef"]
    tc_cut    = bg.get("tc_cutoff", TC_CUTOFF)
    n_targets = len(index)
    results   = []

    for tid, tinfo in index.items():
        fps = tinfo.get("fps")
        if fps is None or len(fps) == 0:
            continue

        tcs   = _tc_batch(query_fp, fps)
        above = tcs[tcs >= tc_cut]
        if len(above) == 0:
            continue

        RS   = float(above.sum())
        N    = len(fps)
        mu   = mu_slope  * N
        beta = max(beta_coef * (N ** 0.5), 1e-9)

        p    = float(gumbel_r.sf(RS, loc=mu, scale=beta))
        e    = max(p * n_targets, 1e-300)       # floor at tiny positive

        if e > e_cutoff:
            continue

        results.append({
            "target_chembl_id" : tid,
            "target_name"      : tinfo["name"],
            "target_type"      : tinfo["target_type"],
            "organism"         : tinfo["organism"],
            "e_value"          : e,
            "p_value"          : round(p, 8),
            "raw_score"        : round(RS, 4),
            "max_tc"           : round(float(tcs.max()), 3),
            "n_hits"           : int((tcs >= tc_cut).sum()),
            "n_ligands"        : N,
            "chembl_url"       : f"https://www.ebi.ac.uk/chembl/target_report_card/{tid}/",
        })

    results.sort(key=lambda x: x["e_value"])
    results = results[:top_n]

    return {
        "status"             : "OK" if results else "NO_PREDICTIONS",
        "predictions"        : results,
        "n_targets_searched" : n_targets,
        "query_smiles"       : query_smiles,
    }

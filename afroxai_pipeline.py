"""
AfroXai-Derep — End-to-end pipeline for dereplication of African natural products.
Four-layer architecture: Tanimoto + XGBoost + Applicability-Domain + PubChem.
"""
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
import shap

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_COLS = ["mw", "logp", "tpsa", "hbd", "hba",
                   "rotatable_bonds", "aromatic_rings", "heavy_atoms",
                   "rings", "fraction_csp3"]

APP_DOMAIN_THRESHOLD = 0.55
KNOWN_SIM_THRESHOLD  = 0.70
KNOWN_CONF_THRESHOLD = 0.85
POSSIBLY_NOVEL_CONF  = 0.50

_ref_df = None
_ref_fps = None
_ref_descs = None
_model = None
_explainer = None
_normalizer = None
_uncharger = None
_te = None
_feature_names = None


def load_artifacts(ref_parquet, ref_fps_npy, model_json):
    global _ref_df, _ref_fps, _ref_descs, _model, _explainer
    global _normalizer, _uncharger, _te, _feature_names
    df = pd.read_parquet(ref_parquet)
    mask = df["np_class"].notna()
    _ref_df = df[mask].reset_index(drop=True)
    _ref_fps = np.load(ref_fps_npy)[mask.values].astype(np.uint8)
    _ref_descs = _ref_df[DESCRIPTOR_COLS].values.astype(np.float32)
    _model = xgb.XGBClassifier()
    _model.load_model(model_json)
    _explainer = shap.TreeExplainer(_model)
    _normalizer = rdMolStandardize.Normalizer()
    _uncharger = rdMolStandardize.Uncharger()
    _te = rdMolStandardize.TautomerEnumerator()
    _feature_names = (
        [f"q_bit_{i}" for i in range(2048)] +
        [f"m_bit_{i}" for i in range(2048)] +
        [f"d_bit_{i}" for i in range(2048)] +
        ["tanimoto_sim"] +
        [f"d_{c}" for c in DESCRIPTOR_COLS]
    )


def _standardise(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _normalizer.normalize(mol)
        mol = _uncharger.uncharge(mol)
        mol = _te.Canonicalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _features(smiles_std):
    try:
        mol = Chem.MolFromSmiles(smiles_std)
        if mol is None:
            return None, None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)
        descs = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol), Lipinski.NumAromaticRings(mol),
            mol.GetNumHeavyAtoms(), Lipinski.RingCount(mol),
            Descriptors.FractionCSP3(mol)
        ], dtype=np.float32)
        return fp_arr, descs
    except Exception:
        return None, None


def _tanimoto_search(query_fp, top_k):
    query_int = query_fp.astype(np.int32)
    ref_int = _ref_fps.astype(np.int32)
    intersection = ref_int @ query_int
    pop_query = query_int.sum()
    pop_ref = ref_int.sum(axis=1)
    union = pop_query + pop_ref - intersection
    sims = np.where(union > 0, intersection / union, 0).astype(np.float32)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return top_idx, sims[top_idx]


def _build_pair_features(qfp, qdesc, mfp, mdesc, sim):
    return np.concatenate([qfp, mfp, qfp ^ mfp, [sim], np.abs(qdesc - mdesc)]).astype(np.float32)


def _explain(pair_features, top_n=10):
    shap_vals = _explainer.shap_values(pair_features.reshape(1, -1))[0]
    df = pd.DataFrame({
        "feature": _feature_names, "value": pair_features,
        "shap": shap_vals, "abs_shap": np.abs(shap_vals)
    }).sort_values("abs_shap", ascending=False).head(top_n)
    df["direction"] = df["shap"].apply(lambda x: "supports match" if x > 0 else "against match")
    return df[["feature", "value", "shap", "direction"]].reset_index(drop=True)


def _query_pubchem(smiles, timeout=8):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"status": "LOOKUP_FAILED", "cid": None, "url": None, "name": None,
                    "reason": "invalid SMILES for InChIKey"}
        inchikey = inchi.MolToInchiKey(mol)
    except Exception as e:
        return {"status": "LOOKUP_FAILED", "cid": None, "url": None, "name": None, "reason": str(e)}
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 404:
            return {"status": "NOT_FOUND", "cid": None, "url": None, "name": None, "inchikey": inchikey}
        if r.status_code != 200:
            return {"status": "LOOKUP_FAILED", "cid": None, "url": None, "name": None,
                    "reason": f"HTTP {r.status_code}", "inchikey": inchikey}
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {"status": "NOT_FOUND", "cid": None, "url": None, "name": None, "inchikey": inchikey}
        cid = cids[0]
        name = None
        try:
            name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/TXT"
            nr = requests.get(name_url, timeout=timeout)
            if nr.status_code == 200:
                name = nr.text.strip()
        except Exception:
            pass
        return {"status": "FOUND", "cid": cid,
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                "name": name, "inchikey": inchikey}
    except requests.exceptions.Timeout:
        return {"status": "LOOKUP_FAILED", "cid": None, "url": None, "name": None, "reason": "timeout"}
    except Exception as e:
        return {"status": "LOOKUP_FAILED", "cid": None, "url": None, "name": None, "reason": str(e)}



SWISSTARGET_ORGANISMS = [
    "Homo sapiens",
    "Mus musculus",
    "Rattus norvegicus",
    "Danio rerio",
    "Caenorhabditis elegans",
    "Drosophila melanogaster",
    "Saccharomyces cerevisiae",
]


def predict_targets_swisstarget(smiles, organism="Homo sapiens", timeout=30):
    """Query SwissTargetPrediction REST API and return ranked target predictions."""
    import urllib.parse

    smiles_std = _standardise(smiles)
    if smiles_std is None:
        return {"status": "INVALID_SMILES", "predictions": [], "organism": organism}

    try:
        url = "http://www.swisstargetprediction.ch/predict.php"
        payload = {"smiles": smiles_std, "organism": organism}
        r = requests.post(url, data=payload, timeout=timeout)
        if r.status_code != 200:
            return {"status": "API_ERROR", "predictions": [], "organism": organism,
                    "reason": f"HTTP {r.status_code}"}

        data = r.json()
        if not data:
            return {"status": "NO_PREDICTIONS", "predictions": [], "organism": organism}

        predictions = []
        for item in data:
            uid = item.get("Uniprot ID", "")
            predictions.append({
                "target":         item.get("Target", ""),
                "common_name":    item.get("Common name", ""),
                "uniprot_id":     uid,
                "chembl_id":      item.get("ChEMBL ID", ""),
                "target_class":   item.get("Target Class", ""),
                "probability":    float(item.get("Probability", 0)),
                "known_actives":  item.get("Known actives (3D/2D)", ""),
                "uniprot_url":    f"https://www.uniprot.org/uniprot/{uid}" if uid else "",
            })

        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return {
            "status":       "OK",
            "predictions":  predictions,
            "organism":     organism,
            "query_smiles": smiles_std,
        }

    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "predictions": [], "organism": organism,
                "reason": "SwissTargetPrediction API timed out"}
    except Exception as e:
        return {"status": "ERROR", "predictions": [], "organism": organism, "reason": str(e)}


def predict_targets(smiles, similarity_threshold=70, top_n=10, timeout=15):
    """
    Predict likely protein targets via ChEMBL similarity API.
    
    Strategy:
      1. Find compounds in ChEMBL similar to the query (Tanimoto >= threshold/100)
      2. For each similar compound, retrieve its annotated bioactivity targets
      3. Aggregate by target frequency and best activity
    
    Args:
        smiles: query SMILES string
        similarity_threshold: ChEMBL similarity threshold (40-100, default 70)
        top_n: number of top targets to return
        timeout: HTTP timeout in seconds
    
    Returns:
        dict with status and predictions list
    """
    import urllib.parse
    
    # Standardise the query first
    smiles_std = _standardise(smiles)
    if smiles_std is None:
        return {"status": "INVALID_SMILES", "predictions": [], "n_similar_compounds": 0}
    
    # ChEMBL similarity API: returns compounds similar to query
    encoded = urllib.parse.quote(smiles_std, safe='')
    sim_url = (
        f"https://www.ebi.ac.uk/chembl/api/data/similarity/{encoded}/"
        f"{int(similarity_threshold)}.json?limit=50"
    )
    
    try:
        r = requests.get(sim_url, timeout=timeout)
        if r.status_code != 200:
            return {"status": "API_ERROR", "predictions": [],
                    "n_similar_compounds": 0,
                    "reason": f"ChEMBL similarity HTTP {r.status_code}"}
        
        sim_data = r.json()
        molecules = sim_data.get("molecules", [])
        
        if not molecules:
            return {"status": "NO_SIMILAR_COMPOUNDS", "predictions": [],
                    "n_similar_compounds": 0,
                    "message": f"No ChEMBL compounds with similarity >= {similarity_threshold}%"}
        
        # For each similar compound, get its bioactivity targets
        target_aggregates = {}  # target_chembl_id -> {name, type, organism, count, max_sim}
        
        for mol in molecules[:25]:  # cap at 25 most-similar to keep it fast
            chembl_id = mol.get("molecule_chembl_id")
            similarity = float(mol.get("similarity", 0))
            if not chembl_id:
                continue
            
            act_url = (
                f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
                f"?molecule_chembl_id={chembl_id}"
                f"&pchembl_value__gte=5"  # active compounds only (pIC50/pKi >= 5)
                f"&limit=20"
            )
            try:
                ar = requests.get(act_url, timeout=timeout)
                if ar.status_code != 200:
                    continue
                activities = ar.json().get("activities", [])
            except Exception:
                continue
            
            for act in activities:
                target_id = act.get("target_chembl_id")
                target_name = act.get("target_pref_name") or "(unnamed target)"
                target_org = act.get("target_organism") or "(unknown organism)"
                if not target_id:
                    continue
                
                if target_id not in target_aggregates:
                    target_aggregates[target_id] = {
                        "target_chembl_id": target_id,
                        "target_name": target_name,
                        "target_organism": target_org,
                        "evidence_count": 0,
                        "max_similarity": 0.0,
                        "supporting_compounds": set(),
                    }
                
                agg = target_aggregates[target_id]
                agg["evidence_count"] += 1
                agg["max_similarity"] = max(agg["max_similarity"], similarity)
                agg["supporting_compounds"].add(chembl_id)
        
        # Rank by combined score: weight similarity higher than evidence count
        predictions = []
        for tid, agg in target_aggregates.items():
            score = (agg["max_similarity"] / 100.0) * 0.7 + min(agg["evidence_count"] / 10.0, 1.0) * 0.3
            predictions.append({
                "target_chembl_id": tid,
                "target_name": agg["target_name"],
                "organism": agg["target_organism"],
                "supporting_compounds": len(agg["supporting_compounds"]),
                "evidence_count": agg["evidence_count"],
                "max_similarity": round(agg["max_similarity"], 1),
                "score": round(score, 3),
                "chembl_url": f"https://www.ebi.ac.uk/chembl/target_report_card/{tid}/",
            })
        
        predictions.sort(key=lambda x: x["score"], reverse=True)
        predictions = predictions[:top_n]
        
        # Build SwissTargetPrediction deep-link (prefills SMILES on their site)
        swisstarget_url = (
            "http://www.swisstargetprediction.ch/result.php?"
            f"smiles={urllib.parse.quote(smiles_std)}&organism=Homo_sapiens"
        )
        
        return {
            "status": "OK",
            "predictions": predictions,
            "n_similar_compounds": len(molecules),
            "swisstarget_url": swisstarget_url,
            "query_smiles": smiles_std,
        }
    
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "predictions": [], "n_similar_compounds": 0,
                "reason": "ChEMBL API timeout"}
    except Exception as e:
        return {"status": "ERROR", "predictions": [], "n_similar_compounds": 0,
                "reason": str(e)}


def dereplicate(query_smiles, top_k=10, check_pubchem=True):
    """Full four-layer dereplication."""
    smiles_std = _standardise(query_smiles)
    if smiles_std is None:
        return {"query_smiles": query_smiles, "standardised_smiles": None,
                "verdict": "INVALID", "verdict_score": None, "top_similarity": None,
                "top_matches": None, "best_match_explanation": None,
                "query_features": None, "pubchem": {"status": "SKIPPED", "cid": None, "url": None, "name": None}}
    
    query_fp, query_desc = _features(smiles_std)
    if query_fp is None:
        return {"query_smiles": query_smiles, "standardised_smiles": smiles_std,
                "verdict": "INVALID", "verdict_score": None, "top_similarity": None,
                "top_matches": None, "best_match_explanation": None,
                "query_features": None, "pubchem": {"status": "SKIPPED", "cid": None, "url": None, "name": None}}
    
    top_idx, top_sims = _tanimoto_search(query_fp, top_k)
    pair_features_batch = np.array([
        _build_pair_features(query_fp, query_desc, _ref_fps[idx], _ref_descs[idx], sim)
        for idx, sim in zip(top_idx, top_sims)
    ])
    confidence_scores = _model.predict_proba(pair_features_batch)[:, 1]
    best_explanation = _explain(pair_features_batch[0])
    
    verdict_score = float(confidence_scores.max())
    top_similarity = float(top_sims.max())
    
    if top_similarity < APP_DOMAIN_THRESHOLD:
        verdict = "NOVEL" if verdict_score < 0.30 else "NOVEL_OR_OUT_OF_DOMAIN"
    elif verdict_score >= KNOWN_CONF_THRESHOLD and top_similarity >= KNOWN_SIM_THRESHOLD:
        verdict = "KNOWN"
    elif verdict_score >= POSSIBLY_NOVEL_CONF:
        verdict = "POSSIBLY_NOVEL"
    else:
        verdict = "NOVEL"
    
    pubchem_result = {"status": "SKIPPED", "cid": None, "url": None, "name": None}
    verdict_note = None
    if check_pubchem:
        pubchem_result = _query_pubchem(smiles_std)
        if pubchem_result["status"] == "FOUND":
            if verdict in ("NOVEL", "NOVEL_OR_OUT_OF_DOMAIN", "POSSIBLY_NOVEL"):
                original = verdict
                verdict = "KNOWN_VIA_PUBCHEM"
                verdict_note = (f"Internal pipeline said {original}, but PubChem CID "
                               f"{pubchem_result['cid']} exists. Compound documented in broader literature.")
            else:
                verdict_note = f"Confirmed in PubChem (CID {pubchem_result['cid']})."
        elif pubchem_result["status"] == "NOT_FOUND":
            if verdict in ("NOVEL", "NOVEL_OR_OUT_OF_DOMAIN"):
                verdict = "NOVEL_CONFIRMED"
                verdict_note = ("Not in Zanthoxylum reference AND not in PubChem. "
                               "Strong novelty signal — worth experimental investigation.")
            elif verdict == "POSSIBLY_NOVEL":
                verdict_note = "Internal POSSIBLY_NOVEL; PubChem also has no record."
            elif verdict == "KNOWN":
                verdict_note = ("Internal says KNOWN but PubChem has no exact-InChIKey match. "
                               "May be stereoisomer or tautomer — manual review recommended.")
        else:
            verdict_note = f"PubChem lookup failed ({pubchem_result.get('reason', 'unknown')})."
    
    top_matches_df = pd.DataFrame({
        "rank": range(1, len(top_idx) + 1),
        "reference_id": _ref_df.iloc[top_idx]["source_id"].values,
        "name": _ref_df.iloc[top_idx]["name"].values,
        "np_class": _ref_df.iloc[top_idx]["np_class"].values,
        "organisms": _ref_df.iloc[top_idx]["organisms"].values,
        "tanimoto_similarity": top_sims.round(3),
        "match_confidence": confidence_scores.round(3),
    })
    
    return {
        "query_smiles": query_smiles,
        "standardised_smiles": smiles_std,
        "verdict": verdict,
        "verdict_score": round(verdict_score, 3),
        "top_similarity": round(top_similarity, 3),
        "top_matches": top_matches_df,
        "best_match_explanation": best_explanation,
        "query_features": dict(zip(DESCRIPTOR_COLS, query_desc.round(3).tolist())),
        "pubchem": pubchem_result,
        "verdict_note": verdict_note,
    }

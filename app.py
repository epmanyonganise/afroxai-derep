"""
AfroXai-Derep — Streamlit web app for explainable dereplication
of Zanthoxylum chalybeum natural products.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw

from afroxai_pipeline import load_artifacts, dereplicate

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="AfroXai-Derep",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-title {
        font-family: Georgia, serif;
        color: #2C5F2D;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6B6B6B;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .verdict-known       { background: #2C5F2D; color: white; }
    .verdict-pubchem     { background: #6E8C3D; color: white; }
    .verdict-possibly    { background: #B8531A; color: white; }
    .verdict-novel       { background: #97BC62; color: #1F2A1F; }
    .verdict-confirmed   { background: #1F2A1F; color: #97BC62; }
    .verdict-ood         { background: #6B6B6B; color: white; }
    .verdict-invalid     { background: #d32f2f; color: white; }
    .verdict-label {
        font-family: Georgia, serif;
        font-size: 1.6rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .metric-card {
        background: #F7F4EC;
        padding: 0.7rem;
        border-radius: 6px;
        text-align: center;
    }
    .metric-value {
        font-family: Georgia, serif;
        font-size: 1.4rem;
        font-weight: bold;
        color: #2C5F2D;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6B6B6B;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Load pipeline artifacts (cached)
# =============================================================================
@st.cache_resource(show_spinner="Loading model and reference data...")
def initialise_pipeline():
    artifacts_dir = Path(__file__).parent / "artifacts"
    load_artifacts(
        ref_parquet=artifacts_dir / "zanthoxylum_features.parquet",
        ref_fps_npy=artifacts_dir / "zanthoxylum_ecfp4.npy",
        model_json=artifacts_dir / "tanimoto_confidence_xgboost.json",
    )
    return True

initialise_pipeline()

# =============================================================================
# Helper functions
# =============================================================================
def render_molecule(smiles, size=(380, 380)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def verdict_styling(verdict):
    return {
        "KNOWN":               ("verdict-known",     "✓",  "Confident match within Zanthoxylum reference"),
        "KNOWN_VIA_PUBCHEM":   ("verdict-pubchem",   "ℹ",  "Documented in broader chemical literature (PubChem)"),
        "POSSIBLY_NOVEL":      ("verdict-possibly",  "⚠",  "Modest similarity — manual review recommended"),
        "NOVEL":               ("verdict-novel",     "★",  "Low similarity, low confidence — possible novel compound"),
        "NOVEL_CONFIRMED":     ("verdict-confirmed", "★★", "Not in Zanthoxylum reference AND not in PubChem"),
        "NOVEL_OR_OUT_OF_DOMAIN": ("verdict-ood",    "?",  "Outside model's training distribution"),
        "INVALID":             ("verdict-invalid",   "✗",  "SMILES could not be parsed"),
    }.get(verdict, ("verdict-novel", "?", "Unknown verdict"))

def shap_chart(explanation_df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2C5F2D" if d == "supports match" else "#B8531A"
              for d in explanation_df["direction"]]
    y_pos = range(len(explanation_df))
    ax.barh(y_pos, explanation_df["shap"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(explanation_df["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on match confidence)")
    ax.set_title("Top features driving the verdict")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<h1 class="main-title">🌿 AfroXai-Derep</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Explainable dereplication & novelty discovery for '
    "<i>Zanthoxylum chalybeum</i> phytochemistry</p>",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "**AfroXai-Derep** combines four layers of evidence to classify "
        "compounds isolated from Knobwood (*Zanthoxylum chalybeum*) extracts:\n\n"
        "1. **Tanimoto similarity** to a 2,294-compound *Zanthoxylum* reference\n"
        "2. **XGBoost confidence layer** (MCC 0.988) — calibrated match trust\n"
        "3. **Applicability-domain check** — rejects out-of-domain queries\n"
        "4. **PubChem cross-reference** — checks broader chemical literature\n\n"
        "SHAP attribution explains every verdict in chemically meaningful terms."
    )
    st.markdown("---")
    st.markdown("### Settings")
    top_k = st.slider("Number of top matches to retrieve", 5, 25, 10)
    check_pubchem = st.checkbox("Cross-reference PubChem", value=True)
    st.markdown("---")
    st.caption("Ed Panashe Manyonganise · Harare Institute of Technology · 2026")

# =============================================================================
# QUERY INPUT
# =============================================================================
EXAMPLES = {
    "cis-Piperitol acetate (in Zanthoxylum)": "CC(=O)O[C@H]1C=C(C)CC[C@H]1C(C)C",
    "Quinine (alkaloid, Cinchona)":            "C[C@@H](O)[C@@H]1CC[C@@H]2CN1CC2C(=C)C=C",
    "Aspirin (synthetic, foreign)":            "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (xanthine, foreign)":            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

st.markdown("### Query compound")
col_input, col_btn = st.columns([5, 1])
with col_input:
    smiles_input = st.text_input(
        "SMILES",
        value=st.session_state.get("smiles_input", ""),
        placeholder="e.g. CC(=O)O[C@H]1C=C(C)CC[C@H]1C(C)C",
        label_visibility="collapsed",
    )
with col_btn:
    analyze = st.button("Analyse", type="primary", width="stretch")

st.markdown("**Try an example:**")
example_cols = st.columns(len(EXAMPLES))
for col, (label, smi) in zip(example_cols, EXAMPLES.items()):
    if col.button(label, width="stretch"):
        st.session_state["smiles_input"] = smi
        st.rerun()

# =============================================================================
# RESULTS — using session_state so results persist across button clicks
# =============================================================================

if analyze and smiles_input:
    with st.spinner("Running dereplication pipeline..."):
        st.session_state["last_result"] = dereplicate(
            smiles_input, top_k=top_k, check_pubchem=check_pubchem
        )
        st.session_state.pop("last_target_result", None)
elif analyze and not smiles_input:
    st.warning("Please enter a SMILES string or click an example button.")

result = st.session_state.get("last_result")

if result is None:
    st.info(
        "👆 Paste a SMILES string above and click **Analyse**, "
        "or click one of the example buttons to see the pipeline in action."
    )
elif result["verdict"] == "INVALID":
    st.error("❌ The SMILES could not be parsed. Please check the input.")
else:
    col_verdict, col_structure = st.columns([3, 2])

    with col_verdict:
        css_class, emoji, description = verdict_styling(result["verdict"])
        st.markdown(
            f'<div class="verdict-box {css_class}">'
            f'<div style="font-size:2rem;">{emoji}</div>'
            f'<div class="verdict-label">{result["verdict"]}</div>'
            f'<div style="margin-top:0.4rem;font-size:0.9rem;opacity:0.95;">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-grid">'
            f'  <div class="metric-card">'
            f'    <div class="metric-value">{result["top_similarity"]}</div>'
            f'    <div class="metric-label">Top Tanimoto similarity</div>'
            f'  </div>'
            f'  <div class="metric-card">'
            f'    <div class="metric-value">{result["verdict_score"]}</div>'
            f'    <div class="metric-label">Match confidence</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        pc = result["pubchem"]
        if pc["status"] == "FOUND":
            st.success(
                f"📘 **PubChem CID {pc['cid']}** — {pc.get('name', '(no IUPAC name)')}  \n"
                f"[Open in PubChem ↗]({pc['url']})"
            )
        elif pc["status"] == "NOT_FOUND":
            st.info("📘 No PubChem record found for this exact structure.")
        elif pc["status"] == "SKIPPED":
            st.caption("PubChem check was skipped.")
        else:
            st.warning(f"PubChem lookup failed: {pc.get('reason', 'unknown')}")
        if result.get("verdict_note"):
            st.caption(f"ℹ️ {result['verdict_note']}")

    with col_structure:
        st.markdown("**Standardised structure**")
        mol_img = render_molecule(result["standardised_smiles"])
        if mol_img:
            st.image(mol_img, width="stretch")
        else:
            st.warning("Could not render structure.")
        st.caption(f"`{result['standardised_smiles']}`")

    tab_matches, tab_shap, tab_props, tab_targets = st.tabs(
        ["🔍 Top matches", "📊 SHAP explanation", "🧪 Query properties", "🎯 Bioactivity targets"]
    )

    with tab_matches:
        st.markdown(f"**Top {len(result['top_matches'])} structurally similar compounds in *Zanthoxylum* reference:**")
        df = result["top_matches"].copy()
        df["organisms"] = df["organisms"].astype(str).apply(
            lambda s: s[:80] + "…" if len(s) > 80 else s
        )
        st.dataframe(df, width="stretch", hide_index=True)

    with tab_shap:
        st.markdown("**Top features driving the model's verdict on the best match.**")
        st.markdown(
            "<small>Green bars support a match (push toward KNOWN); "
            "orange bars argue against it (push toward NOVEL).</small>",
            unsafe_allow_html=True,
        )
        fig = shap_chart(result["best_match_explanation"])
        st.pyplot(fig, width="stretch")
        with st.expander("Show feature attribution table"):
            st.dataframe(result["best_match_explanation"], width="stretch", hide_index=True)

    with tab_props:
        st.markdown("**Computed physicochemical properties of the query:**")
        props = result["query_features"]
        prop_df = pd.DataFrame([
            {"Property": "Molecular Weight (Da)",  "Value": f"{props['mw']:.2f}"},
            {"Property": "LogP",                    "Value": f"{props['logp']:.2f}"},
            {"Property": "TPSA (Å²)",               "Value": f"{props['tpsa']:.2f}"},
            {"Property": "H-bond donors",           "Value": f"{int(props['hbd'])}"},
            {"Property": "H-bond acceptors",        "Value": f"{int(props['hba'])}"},
            {"Property": "Rotatable bonds",         "Value": f"{int(props['rotatable_bonds'])}"},
            {"Property": "Aromatic rings",          "Value": f"{int(props['aromatic_rings'])}"},
            {"Property": "Total rings",             "Value": f"{int(props['rings'])}"},
            {"Property": "Heavy atoms",             "Value": f"{int(props['heavy_atoms'])}"},
            {"Property": "Fraction sp3",            "Value": f"{props['fraction_csp3']:.2f}"},
        ])
        st.dataframe(prop_df, width="stretch", hide_index=True)

    with tab_targets:
        st.markdown("**Predicted protein targets**")
        st.markdown(
            "<small>Predict likely biological targets by aggregating annotated activities "
            "of structurally similar known ligands from the ChEMBL database. "
            "Useful for generating bioactivity hypotheses on novel compounds.</small>",
            unsafe_allow_html=True,
        )

        col_sim, col_top = st.columns(2)
        with col_sim:
            sim_threshold = st.slider(
                "ChEMBL similarity threshold (%)",
                min_value=40, max_value=95, value=70, step=5,
                help="Higher values = stricter similarity, fewer but more confident predictions",
                key="target_sim_threshold",
            )
        with col_top:
            target_top_n = st.slider(
                "Number of top targets to show",
                min_value=5, max_value=20, value=10,
                key="target_top_n",
            )

        if st.button("🎯 Predict protein targets", type="primary", width="stretch"):
            with st.spinner("Querying ChEMBL similarity API and aggregating bioactivities..."):
                from afroxai_pipeline import predict_targets
                st.session_state["last_target_result"] = predict_targets(
                    result["standardised_smiles"],
                    similarity_threshold=sim_threshold,
                    top_n=target_top_n,
                )

        target_result = st.session_state.get("last_target_result")
        if target_result is not None:
            status = target_result["status"]

            if status == "OK" and target_result["predictions"]:
                n_sim = target_result["n_similar_compounds"]
                n_targets = len(target_result["predictions"])
                st.success(
                    f"✅ Found **{n_sim}** structurally similar ChEMBL compounds, "
                    f"aggregated to **{n_targets}** predicted targets."
                )

                pred_df = pd.DataFrame(target_result["predictions"])
                display_df = pd.DataFrame({
                    "Rank": range(1, len(pred_df) + 1),
                    "Target": pred_df["target_name"].astype(str),
                    "Organism": pred_df["organism"].astype(str),
                    "Supporting compounds": pred_df["supporting_compounds"].astype(str),
                    "Max similarity (%)": pred_df["max_similarity"].apply(lambda v: f"{v:.1f}"),
                    "Score": pred_df["score"].apply(lambda v: f"{v:.3f}"),
                })
                st.dataframe(display_df, width="stretch", hide_index=True)

                with st.expander("Open targets in ChEMBL"):
                    for _, row in pred_df.iterrows():
                        st.markdown(
                            f"- **{row['target_name']}** ({row['organism']}) — "
                            f"[{row['target_chembl_id']}]({row['chembl_url']})"
                        )

                st.markdown("---")
                st.markdown(
                    f"**🔬 Verify with SwissTargetPrediction**  \n"
                    f"For independent verification using the canonical SwissTargetPrediction tool:  \n"
                    f"[Open in SwissTargetPrediction ↗]({target_result['swisstarget_url']})"
                )

            elif status == "NO_SIMILAR_COMPOUNDS":
                st.warning(
                    f"No ChEMBL compounds found with similarity ≥ {sim_threshold}%. "
                    "Try lowering the threshold, or this compound may be genuinely novel "
                    "with no near-neighbours in ChEMBL."
                )
                import urllib.parse
                fallback_url = (
                    "http://www.swisstargetprediction.ch/result.php?"
                    f"smiles={urllib.parse.quote(result['standardised_smiles'])}"
                    "&organism=Homo_sapiens"
                )
                st.markdown(
                    f"**🔬 Try SwissTargetPrediction directly:**  \n"
                    f"[Open in SwissTargetPrediction ↗]({fallback_url})"
                )

            elif status == "INVALID_SMILES":
                st.error("The query SMILES could not be standardised for ChEMBL lookup.")

            else:
                reason = target_result.get("reason", "unknown")
                st.error(f"Target prediction failed: {reason}")
                st.caption("ChEMBL API may be temporarily unavailable. Try again in a moment.")
        else:
            st.info(
                "👆 Click **Predict protein targets** to query ChEMBL for bioactivity hypotheses. "
                "This typically takes 10–20 seconds depending on the number of similar compounds."
            )
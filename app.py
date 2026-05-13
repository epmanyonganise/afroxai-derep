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

import time
from collections import Counter
from afroxai_pipeline import (
    load_artifacts, dereplicate, predict_targets,
)
from sea_local import load_sea_index, predict_sea

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
    /*
     * Natural-product palette
     * ─────────────────────────────────────────────────────
     * Forest green  #2D5016   primary / known
     * Sage green    #6B8F3E   secondary / pubchem
     * Amber gold    #C49A2D   caution / possibly novel
     * Young leaf    #A3BE72   novel / hopeful
     * Dark earth    #1A2510   confirmed novel
     * Bark brown    #7A6B52   out-of-domain / neutral
     * Parchment     #F5F0E3   card backgrounds
     * Stone         #6B6355   muted text
     */
    .main-title {
        font-family: Georgia, serif;
        color: #2D5016;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #7A6B52;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .verdict-known       { background: #2D5016; color: #F5F0E3; }
    .verdict-pubchem     { background: #6B8F3E; color: #F5F0E3; }
    .verdict-possibly    { background: #C49A2D; color: #1A2510; }
    .verdict-novel       { background: #A3BE72; color: #1A2510; }
    .verdict-confirmed   { background: #1A2510; color: #A3BE72; }
    .verdict-ood         { background: #7A6B52; color: #F5F0E3; }
    .verdict-invalid     { background: #8B3020; color: #F5F0E3; }
    /* ── input card ── */
    .stp-card-title {
        font-weight: 600;
        font-size: 1rem;
        text-align: center;
        color: #2D5016;
        padding: 0.55rem 0 0.75rem 0;
        border-bottom: 2px solid #A3BE72;
        margin-bottom: 0.9rem;
    }
    .stp-analyse-caption {
        text-align: center;
        color: #7A6B52;
        font-style: italic;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }
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
        background: #F5F0E3;
        padding: 0.7rem;
        border-radius: 6px;
        text-align: center;
        border-left: 3px solid #A3BE72;
    }
    .metric-value {
        font-family: Georgia, serif;
        font-size: 1.4rem;
        font-weight: bold;
        color: #2D5016;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #7A6B52;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-hint {
        font-size: 0.7rem;
        color: #7A6B52;
        font-style: italic;
        margin-top: 0.2rem;
        line-height: 1.3;
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

@st.cache_resource(show_spinner=False)
def load_sea_payload():
    idx = Path(__file__).parent / "artifacts" / "sea_index.pkl.gz"
    if not idx.exists():
        return None
    try:
        return load_sea_index(idx)
    except Exception:
        return None

_sea_payload = load_sea_payload()

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
        "KNOWN": (
            "verdict-known", "✓",
            "This compound closely matches one already documented in our Zanthoxylum reference library. "
            "High structural similarity and strong model confidence both point to a known constituent.",
        ),
        "KNOWN_VIA_PUBCHEM": (
            "verdict-pubchem", "ℹ",
            "Not in our internal Zanthoxylum database, but found in PubChem — the world's largest open chemical library. "
            "The compound is documented in the broader scientific literature.",
        ),
        "POSSIBLY_NOVEL": (
            "verdict-possibly", "⚠",
            "Moderate structural similarity to known compounds, but not a confident match. "
            "Could be a known compound with slight variation, or genuinely new — experimental confirmation is recommended.",
        ),
        "NOVEL": (
            "verdict-novel", "★",
            "Low structural similarity to anything in the reference library, and the model has low confidence. "
            "This compound may be new to Zanthoxylum phytochemistry — worth further investigation.",
        ),
        "NOVEL_CONFIRMED": (
            "verdict-confirmed", "★★",
            "Strong novelty signal: the compound is not structurally similar to any reference compound AND has no PubChem record. "
            "Prioritise for full structural characterisation and biological evaluation.",
        ),
        "NOVEL_OR_OUT_OF_DOMAIN": (
            "verdict-ood", "?",
            "The compound is so different from the model's training data that a reliable verdict cannot be given. "
            "The model has not seen enough similar structures to make a confident call — treat results with caution.",
        ),
        "INVALID": (
            "verdict-invalid", "✗",
            "The SMILES string could not be interpreted as a valid chemical structure. Please check the input.",
        ),
    }.get(verdict, ("verdict-novel", "?", "Unknown verdict"))

_SHAP_NAMES = {
    "tanimoto_sim":       "overall structural similarity (Tanimoto)",
    "d_mw":              "molecular weight",
    "d_logp":            "lipophilicity (LogP)",
    "d_tpsa":            "polar surface area",
    "d_hbd":             "hydrogen-bond donors",
    "d_hba":             "hydrogen-bond acceptors",
    "d_rotatable_bonds": "rotatable bonds",
    "d_aromatic_rings":  "aromatic ring count",
    "d_heavy_atoms":     "heavy atom count",
    "d_rings":           "ring count",
    "d_fraction_csp3":   "fraction of saturated carbons",
}

def shap_plain_english(result):
    expl  = result["best_match_explanation"]
    verd  = result["verdict"]
    best  = result["top_matches"].iloc[0]
    sim   = result["top_similarity"]
    conf  = result["verdict_score"]

    is_desc  = expl["feature"].isin(_SHAP_NAMES)
    bit_expl = expl[~is_desc]
    desc_expl = expl[is_desc]
    n_sup = int((bit_expl["direction"] == "supports match").sum())
    n_agt = int((bit_expl["direction"] == "against match").sum())

    OPENINGS = {
        "KNOWN": (
            f"✅ **Confident match — known *Zanthoxylum* constituent.**  \n"
            f"Best matches **{best['name']}** (NP class: *{best['np_class']}*) "
            f"with Tanimoto similarity **{sim}** and model confidence **{conf}**."
        ),
        "KNOWN_VIA_PUBCHEM": (
            f"📘 **Match confirmed via PubChem.**  \n"
            f"Not in the internal reference but documented in broader literature. "
            f"Closest internal match: **{best['name']}** (similarity {sim}, confidence {conf})."
        ),
        "POSSIBLY_NOVEL": (
            f"⚠️ **Possibly novel — manual review advised.**  \n"
            f"Moderate structural similarity (Tanimoto {sim}) to **{best['name']}**. "
            f"Borderline confidence ({conf}); experimental confirmation recommended."
        ),
        "NOVEL": (
            f"🔬 **Likely novel compound.**  \n"
            f"Best internal match (**{best['name']}**) scores only {sim} Tanimoto similarity "
            f"and model confidence is low ({conf}). "
            f"This may represent a new chemical entity from *Zanthoxylum chalybeum*."
        ),
        "NOVEL_CONFIRMED": (
            f"⭐ **Novel — high confidence.**  \n"
            f"Low structural similarity ({sim}) to the reference database AND absent from PubChem. "
            f"Strongest novelty signal — prioritise for full structural elucidation."
        ),
        "NOVEL_OR_OUT_OF_DOMAIN": (
            f"❓ **Outside model's reliable range.**  \n"
            f"Tanimoto similarity ({sim}) is very low — the model has not been trained on closely "
            f"related structures. Interpret with caution."
        ),
    }
    parts = [OPENINGS.get(verd, f"**Verdict: {verd}** — similarity {sim}, confidence {conf}."), ""]

    parts.append(
        f"**Structural fingerprints:** {n_sup} substructural features match the reference; "
        f"{n_agt} features differ."
    )
    sup_d = desc_expl[desc_expl["direction"] == "supports match"]
    agt_d = desc_expl[desc_expl["direction"] == "against match"]
    if not sup_d.empty:
        names = [_SHAP_NAMES.get(f, f) for f in sup_d["feature"].head(3)]
        parts.append(f"**Matching physicochemical properties:** {', '.join(names)}.")
    if not agt_d.empty:
        names = [_SHAP_NAMES.get(f, f) for f in agt_d["feature"].head(3)]
        parts.append(f"**Differing physicochemical properties:** {', '.join(names)} deviate from the reference.")
    return "\n\n".join(parts)

def target_class_piechart(predictions):
    classes = [p.get("target_class") or p.get("target_type") or "Other" for p in predictions]
    counts = Counter(classes)
    if not counts:
        return None
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*items)
    palette = ["#2D5016","#6B8F3E","#A3BE72","#C49A2D","#7A6B52",
               "#1A2510","#4A7B5F","#8B5E3C","#3A5F8B","#C49A2D"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%", startangle=140,
        colors=palette[:len(labels)], pctdistance=0.80,
        wedgeprops=dict(width=0.55),
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title("Target Classes", fontfamily="Georgia", fontsize=11, pad=8)
    plt.tight_layout()
    return fig

def fmt_evalue(e: float) -> str:
    """Human-readable E-value: scientific for small, decimal for moderate."""
    if e < 1e-4:
        return f"{e:.2e}"
    if e < 0.1:
        return f"{e:.5f}"
    return f"{e:.3f}"

def sea_significance(e: float) -> str:
    if e < 0.01:  return "High ✓✓"
    if e < 1.0:   return "Moderate ✓"
    if e < 10.0:  return "Weak"
    return "—"

def shap_chart(explanation_df):
    import plotly.graph_objects as go

    def _feat_label(f):
        return _SHAP_NAMES.get(f, f)

    def _feat_hint(row):
        f = row["feature"]
        readable = _SHAP_NAMES.get(f, None)
        if readable:
            return readable
        if f.startswith("q_bit_") or f.startswith("m_bit_"):
            return f"Fingerprint substructure bit {f.split('_')[-1]}"
        return f

    colors     = ["#2D5016" if d == "supports match" else "#C49A2D"
                  for d in explanation_df["direction"]]
    y_labels   = [_feat_label(f) for f in explanation_df["feature"]]
    hover_texts = [
        (
            f"<b>{_feat_hint(row)}</b><br>"
            f"SHAP value: {row['shap']:+.4f}<br>"
            f"Effect: {row['direction']}"
        )
        for _, row in explanation_df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=explanation_df["shap"],
        y=y_labels,
        orientation="h",
        marker_color=colors,
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(bgcolor="white", font_size=13, font_color="black"),
    ))
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title=dict(text="Top features driving the verdict", font_family="Georgia, serif", font_size=15, font_color="black"),
        xaxis_title="SHAP value  (positive = supports KNOWN, negative = supports NOVEL)",
        xaxis=dict(tickfont=dict(color="black", size=11), title_font=dict(color="black")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="black", size=12)),
        height=max(380, len(explanation_df) * 32),
        plot_bgcolor="#F5F0E3",
        paper_bgcolor="white",
        margin=dict(l=10, r=20, t=50, b=50),
        font=dict(family="Georgia, serif", color="black"),
    )
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
    st.markdown("### Settings")
    top_k = st.slider("Number of top matches to retrieve", 5, 25, 10)
    check_pubchem = st.checkbox("Cross-reference PubChem", value=True)
    st.markdown("---")
    st.markdown("### SEA Target Prediction")
    if _sea_payload:
        _n_sea = len(_sea_payload["index"])
        st.success(f"Index loaded — {_n_sea:,} targets")
    else:
        st.warning("Index not built yet.")
        st.markdown(
            "Run **once** to enable SEA predictions:  \n"
            "```\npython build_sea_index.py --fast\n```\n"
            "*(~10 min, pChEMBL ≥ 6)*"
        )
    st.markdown("---")
    st.caption("Ed Panashe Manyonganise · Harare Institute of Technology · 2026")

# =============================================================================
# QUERY INPUT  — STP-style: bordered card with Ketcher molecule editor
# =============================================================================
from streamlit_ketcher import st_ketcher

EXAMPLES = {
    "cis-Piperitol acetate (Zanthoxylum)": "CC(=O)O[C@H]1C=C(C)CC[C@H]1C(C)C",
    "Quinine (Cinchona alkaloid)":         "C[C@@H](O)[C@@H]1CC[C@@H]2CN1CC2C(=C)C=C",
    "Aspirin (synthetic)":                 "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (xanthine)":                 "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

if "ketcher_id"    not in st.session_state: st.session_state["ketcher_id"]    = 0
if "ketcher_smi"   not in st.session_state: st.session_state["ketcher_smi"]   = ""
if "main_smi_input" not in st.session_state: st.session_state["main_smi_input"] = ""

def _on_smi_text_change():
    new_smi = st.session_state.get("main_smi_input", "").strip()
    if new_smi != st.session_state["ketcher_smi"]:
        st.session_state["ketcher_smi"] = new_smi
        st.session_state["ketcher_id"] += 1

def _load_main_example():
    choice = st.session_state.get("main_ex_select", "Examples")
    if choice != "Examples" and choice in EXAMPLES:
        smi = EXAMPLES[choice]
        st.session_state["ketcher_smi"]    = smi
        st.session_state["main_smi_input"] = smi
        st.session_state["ketcher_id"] += 1

with st.container(border=True):
    st.markdown(
        '<div class="stp-card-title">Paste a SMILES in this box, or draw a molecule</div>',
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.text_input(
            "SMILES",
            label_visibility="collapsed",
            placeholder="e.g. CC(=O)O[C@H]1C=C(C)CC[C@H]1C(C)C",
            key="main_smi_input",
            on_change=_on_smi_text_change,
        )
        ex_col, cl_col = st.columns([4, 1])
        with ex_col:
            st.selectbox(
                "Examples",
                ["Examples"] + list(EXAMPLES.keys()),
                label_visibility="collapsed",
                key="main_ex_select",
                on_change=_load_main_example,
            )
        with cl_col:
            if st.button("Clear", key="main_clear"):
                st.session_state["ketcher_smi"]    = ""
                st.session_state["main_smi_input"] = ""
                st.session_state["ketcher_id"] += 1
                st.rerun()

    with col_right:
        drawn_smi = st_ketcher(
            value=st.session_state["ketcher_smi"],
            height=380,
            key=f"ketcher_{st.session_state['ketcher_id']}",
        )
        # When user draws, sync back to text box and state
        if drawn_smi and drawn_smi != st.session_state["ketcher_smi"]:
            st.session_state["ketcher_smi"]    = drawn_smi
            st.session_state["main_smi_input"] = drawn_smi

main_smi = st.session_state.get("main_smi_input", "").strip() or st.session_state.get("ketcher_smi", "")

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("Analyse", type="primary", width="stretch", key="btn_analyse")
    st.markdown(
        '<p class="stp-analyse-caption">(Can take a few seconds)</p>',
        unsafe_allow_html=True,
    )

# =============================================================================
# RESULTS — using session_state so results persist across button clicks
# =============================================================================

if analyze and main_smi:
    _prog = st.progress(0, text="Standardising SMILES…")

    def _on_progress(pct, msg=""):
        _prog.progress(pct, text=msg)

    _result = dereplicate(
        main_smi, top_k=top_k, check_pubchem=check_pubchem,
        on_progress=_on_progress,
    )
    _sea    = {"status": "SKIPPED"}
    _chembl = {"status": "SKIPPED"}

    if _result["verdict"] != "INVALID" and _result.get("standardised_smiles"):
        _std = _result["standardised_smiles"]
        if _sea_payload:
            _prog.progress(94, text="Running SEA target prediction (local)…")
            _sea = predict_sea(_std, _sea_payload, top_n=20)
        else:
            # No SEA index — fall back to ChEMBL API
            _prog.progress(94, text="Predicting protein targets (ChEMBL)…")
            _chembl = predict_targets(_std, similarity_threshold=70, top_n=15)

    _prog.progress(100, text="Analysis complete!")
    time.sleep(0.35)
    _prog.empty()

    st.session_state["last_result"]      = _result
    st.session_state["last_sea_auto"]    = _sea
    st.session_state["last_chembl_auto"] = _chembl
    st.session_state.pop("last_chembl_result", None)
elif analyze and not main_smi:
    st.warning("Please paste a SMILES string, select an example, or draw a molecule.")

result      = st.session_state.get("last_result")
sea_auto    = st.session_state.get("last_sea_auto")
chembl_auto = st.session_state.get("last_chembl_auto")

if result is None:
    st.info("Paste a SMILES string above and click **Analyse** to run the pipeline.")
elif result["verdict"] == "INVALID":
    st.error("❌ The SMILES could not be parsed. Please check the input.")
else:
    # ── 1. VERDICT ──────────────────────────────────────────────────────────
    css_class, emoji, description = verdict_styling(result["verdict"])
    st.markdown(
        f'<div class="verdict-box {css_class}">'
        f'<div style="font-size:2rem;">{emoji}</div>'
        f'<div class="verdict-label">{result["verdict"].replace("_", " ")}</div>'
        f'<div style="margin-top:0.5rem;font-size:0.88rem;opacity:0.95;max-width:640px;margin-left:auto;margin-right:auto;line-height:1.5;">{description}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-grid">'
        f'  <div class="metric-card">'
        f'    <div class="metric-value">{result["top_similarity"]}</div>'
        f'    <div class="metric-label">Top Tanimoto Similarity</div>'
        f'    <div class="metric-hint">How structurally similar your compound is to the closest known reference compound.<br>0 = completely different &nbsp;|&nbsp; 1 = identical</div>'
        f'  </div>'
        f'  <div class="metric-card">'
        f'    <div class="metric-value">{result["verdict_score"]}</div>'
        f'    <div class="metric-label">Match Confidence</div>'
        f'    <div class="metric-hint">How certain the AI model is about its verdict, based on multiple molecular features.<br>0 = unsure &nbsp;|&nbsp; 1 = very confident</div>'
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

    # ── 2. PLAIN-ENGLISH EXPLANATION ─────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Why this verdict?**")
        st.markdown(shap_plain_english(result))

    # ── 3. TOP SIMILAR MATCHES ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            f"**Top {len(result['top_matches'])} structurally similar compounds "
            f"in *Zanthoxylum* reference**"
        )
        _df = result["top_matches"].copy()
        _df["organisms"] = _df["organisms"].astype(str).apply(
            lambda s: s[:80] + "…" if len(s) > 80 else s
        )
        st.dataframe(_df, width="stretch", hide_index=True)

    # ── 4. SHAP CHART ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**What drove this verdict?**")
        st.markdown(
            "<small>Forest-green bars = features that <b>support</b> a match to the reference "
            "(push toward KNOWN). &nbsp; Amber bars = features that <b>argue against</b> it "
            "(push toward NOVEL).</small>",
            unsafe_allow_html=True,
        )
        _shap_fig = shap_chart(result["best_match_explanation"])
        st.plotly_chart(_shap_fig, use_container_width=True)
        with st.expander("Show full feature attribution table"):
            st.dataframe(result["best_match_explanation"], width="stretch", hide_index=True)

    # ── 5. QUERY PROPERTIES ───────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Physicochemical properties of your compound**")
        _props = result["query_features"]
        _prop_df = pd.DataFrame([
            {"Property": "Molecular Weight (Da)", "Value": f"{_props['mw']:.2f}",
             "What it means": "Total mass of the molecule — larger molecules are harder to absorb orally"},
            {"Property": "LogP",                  "Value": f"{_props['logp']:.2f}",
             "What it means": "Fat-solubility: negative = water-loving, positive = fat-loving (affects cell penetration)"},
            {"Property": "TPSA (Å²)",             "Value": f"{_props['tpsa']:.2f}",
             "What it means": "Polar surface area — high values generally mean poor membrane permeability"},
            {"Property": "H-bond donors",         "Value": f"{int(_props['hbd'])}",
             "What it means": "N–H and O–H groups that donate H-bonds — too many reduce oral bioavailability"},
            {"Property": "H-bond acceptors",      "Value": f"{int(_props['hba'])}",
             "What it means": "N and O atoms that accept H-bonds — affects solubility and target binding"},
            {"Property": "Rotatable bonds",       "Value": f"{int(_props['rotatable_bonds'])}",
             "What it means": "Molecular flexibility — more bonds means more conformations and affects binding pose"},
            {"Property": "Aromatic rings",        "Value": f"{int(_props['aromatic_rings'])}",
             "What it means": "Flat ring systems — common in alkaloids, flavonoids, and other natural product classes"},
            {"Property": "Total rings",           "Value": f"{int(_props['rings'])}",
             "What it means": "Total ring count including saturated (non-aromatic) rings"},
            {"Property": "Heavy atoms",           "Value": f"{int(_props['heavy_atoms'])}",
             "What it means": "Count of all non-hydrogen atoms — a rough proxy for molecular size"},
            {"Property": "Fraction sp3 carbons",  "Value": f"{_props['fraction_csp3']:.2f}",
             "What it means": "3D shape character — higher values mean a more saturated, three-dimensional scaffold"},
        ])
        st.dataframe(_prop_df, width="stretch", hide_index=True)

    # ── 6. PREDICTED PROTEIN TARGETS ─────────────────────────────────────────
    _sea_ok    = sea_auto    and sea_auto.get("status")    == "OK" and sea_auto.get("predictions")
    _chembl_ok = chembl_auto and chembl_auto.get("status") == "OK" and chembl_auto.get("predictions")

    def _colour_sig(val):
        if "✓✓" in val: return "background-color:#2D5016;color:#F5F0E3"
        if "✓"  in val: return "background-color:#6B8F3E;color:#F5F0E3"
        if val == "Weak": return "background-color:#C49A2D;color:#1A2510"
        return "color:#7A6B52"

    if _sea_ok or _chembl_ok:
        _preds_for_pie = (sea_auto if _sea_ok else chembl_auto).get("predictions", [])
        col_tbl, col_pie = st.columns([3, 2])

        with col_tbl:
            if _sea_ok:
                _preds      = sea_auto["predictions"]
                _n_searched = sea_auto.get("n_targets_searched", "?")
                with st.container(border=True):
                    st.markdown(
                        f'<div class="stp-card-title">SEA Predicted Protein Targets'
                        f' &nbsp;— {len(_preds)} significant hits'
                        f' / {_n_searched:,} targets searched</div>',
                        unsafe_allow_html=True,
                    )
                    _tdf = pd.DataFrame(_preds)
                    _display_tdf = pd.DataFrame({
                        "Target":       _tdf["target_name"],
                        "Target type":  _tdf["target_type"],
                        "E-value":      _tdf["e_value"].apply(fmt_evalue),
                        "Max Tc":       _tdf["max_tc"],
                        "Hits / Set":   _tdf.apply(lambda r: f"{r['n_hits']} / {r['n_ligands']}", axis=1),
                        "Significance": _tdf["e_value"].apply(sea_significance),
                    })
                    st.dataframe(
                        _display_tdf.style.map(_colour_sig, subset=["Significance"]),
                        width="stretch", hide_index=True,
                    )
                    with st.expander("Open targets in ChEMBL"):
                        for _, _row in _tdf.iterrows():
                            st.markdown(
                                f"- **{_row['target_name']}** ({_row['organism']}) — "
                                f"[{_row['target_chembl_id']}]({_row['chembl_url']})"
                            )
            else:
                _preds = chembl_auto["predictions"]
                with st.container(border=True):
                    st.markdown(
                        f'<div class="stp-card-title">Predicted Protein Targets — ChEMBL Similarity'
                        f' &nbsp;({len(_preds)} targets, ≥70% similarity)</div>',
                        unsafe_allow_html=True,
                    )
                    _tdf = pd.DataFrame(_preds)
                    _display_tdf = pd.DataFrame({
                        "Target":             _tdf["target_name"],
                        "Organism":           _tdf["organism"],
                        "Target type":        _tdf["target_type"],
                        "Supporting cpds":    _tdf["supporting_compounds"],
                        "Max similarity (%)": _tdf["max_similarity"].apply(lambda v: f"{v:.1f}"),
                        "Score":              _tdf["score"].round(3),
                    })
                    st.dataframe(
                        _display_tdf.style.background_gradient(subset=["Score"], cmap="YlGn", vmin=0, vmax=1),
                        width="stretch", hide_index=True,
                    )
                    with st.expander("Open targets in ChEMBL"):
                        for _, _row in _tdf.iterrows():
                            st.markdown(
                                f"- **{_row['target_name']}** ({_row['organism']}) — "
                                f"[{_row['target_chembl_id']}]({_row['chembl_url']})"
                            )

        with col_pie:
            with st.container(border=True):
                st.markdown("**Predicted target types**")
                _pie_fig = target_class_piechart(_preds_for_pie)
                if _pie_fig:
                    st.pyplot(_pie_fig, width="stretch")

    elif sea_auto and sea_auto.get("status") == "NO_PREDICTIONS":
        st.info("No SEA target hits above E-value threshold for this compound.")
    elif not _sea_payload:
        st.info(
            "SEA target prediction is not yet enabled.  \n"
            "Run `python build_sea_index.py --fast` once (~10 min) to build the index, "
            "then restart the app."
        )

    # ── 7. ChEMBL MANUAL REQUERY (optional) ──────────────────────────────────
    with st.container(border=True):
        st.markdown("**ChEMBL bioactivity evidence** *(optional — queries live API)*")
        _col_sim, _col_top = st.columns(2)
        with _col_sim:
            sim_threshold = st.slider(
                "Similarity threshold (%)", 40, 95, 70, 5,
                help="Lower = more results, less precise",
                key="target_sim_threshold",
            )
        with _col_top:
            target_top_n = st.slider("Max targets", 5, 25, 15, key="target_top_n")

        if st.button("Re-query ChEMBL", type="secondary", key="btn_chembl_requery"):
            with st.spinner("Querying ChEMBL…"):
                st.session_state["last_chembl_result"] = predict_targets(
                    result["standardised_smiles"],
                    similarity_threshold=sim_threshold,
                    top_n=target_top_n,
                )

        _chembl_requery = st.session_state.get("last_chembl_result")
        if _chembl_requery is not None:
            _cstatus = _chembl_requery["status"]
            if _cstatus == "OK" and _chembl_requery["predictions"]:
                st.success(
                    f"**{_chembl_requery['n_similar_compounds']}** similar compounds → "
                    f"**{len(_chembl_requery['predictions'])}** targets at {sim_threshold}% threshold."
                )
                _rpdf = pd.DataFrame(_chembl_requery["predictions"])
                _rdisp = pd.DataFrame({
                    "Target":             _rpdf["target_name"].astype(str),
                    "Organism":           _rpdf["organism"].astype(str),
                    "Target type":        _rpdf["target_type"].astype(str),
                    "Supporting cpds":    _rpdf["supporting_compounds"],
                    "Max similarity (%)": _rpdf["max_similarity"].apply(lambda v: f"{v:.1f}"),
                    "Score":              _rpdf["score"].round(3),
                })
                st.dataframe(
                    _rdisp.style.background_gradient(subset=["Score"], cmap="YlGn", vmin=0, vmax=1),
                    width="stretch", hide_index=True,
                )
                with st.expander("Open in ChEMBL"):
                    for _, _row in _rpdf.iterrows():
                        st.markdown(
                            f"- **{_row['target_name']}** ({_row['organism']}) — "
                            f"[{_row['target_chembl_id']}]({_row['chembl_url']})"
                        )
            elif _cstatus == "NO_SIMILAR_COMPOUNDS":
                st.warning(f"No ChEMBL compounds at {sim_threshold}% — try lowering the threshold.")
            elif _cstatus == "INVALID_SMILES":
                st.error("SMILES could not be standardised for ChEMBL lookup.")
            else:
                st.error(f"ChEMBL query failed: {_chembl_requery.get('reason', 'unknown')}")
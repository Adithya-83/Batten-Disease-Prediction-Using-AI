import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import cv2
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Batten-AI | Batten Disease Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM UI / INTERACTION LAYER
# ============================================================

st.markdown(
r"""
<style>
/* ---------- Base ---------- */
:root {
    --navy: #0b1f3a;
    --blue: #2563eb;
    --violet: #6d5dfc;
    --cyan: #22d3ee;
    --ink: #17324d;
    --muted: #61758d;
    --line: rgba(148,163,184,.24);
    --glass: rgba(255,255,255,.76);
    --shadow: 0 18px 55px rgba(15,23,42,.10);
}

.stApp {
    background:
        radial-gradient(circle at 8% 4%, rgba(37,99,235,.15), transparent 25%),
        radial-gradient(circle at 94% 7%, rgba(109,93,252,.14), transparent 25%),
        radial-gradient(circle at 52% 105%, rgba(34,211,238,.10), transparent 28%),
        linear-gradient(135deg, #f4f8fc 0%, #edf3fa 48%, #f7f9fc 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.25rem;
    padding-bottom: 5rem;
}

.stApp p, .stApp span, .stApp label {
    color: var(--ink);
}

/* ---------- Hide Streamlit chrome ---------- */
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px 28px;
    border: 1px solid rgba(255,255,255,.70);
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(255,255,255,.84), rgba(242,247,255,.68));
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
    animation: revealUp .7s ease both;
}

.hero:before,
.hero:after {
    content: "";
    position: absolute;
    border-radius: 999px;
    filter: blur(2px);
    pointer-events: none;
}

.hero:before {
    width: 170px;
    height: 170px;
    right: -60px;
    top: -75px;
    background: radial-gradient(circle, rgba(37,99,235,.22), transparent 68%);
    animation: drift 7s ease-in-out infinite;
}

.hero:after {
    width: 140px;
    height: 140px;
    left: -55px;
    bottom: -75px;
    background: radial-gradient(circle, rgba(109,93,252,.18), transparent 68%);
    animation: drift 9s ease-in-out infinite reverse;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 12px;
}

.brand-mark {
    width: 46px;
    height: 46px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    color: white !important;
    font-size: 23px;
    background: linear-gradient(135deg, #173b7a, #2563eb 55%, #6d5dfc);
    box-shadow: 0 10px 28px rgba(37,99,235,.27);
    animation: softFloat 4.5s ease-in-out infinite;
}

.brand-name {
    font-size: 14px;
    font-weight: 800;
    letter-spacing: .14em;
    color: #355a83 !important;
    text-transform: uppercase;
}

.hero h1 {
    color: #0b2344 !important;
    font-size: clamp(31px, 4vw, 48px) !important;
    line-height: 1.04 !important;
    letter-spacing: -1.7px;
    font-weight: 850 !important;
    margin: 0 0 9px !important;
}

.hero-sub {
    max-width: 760px;
    color: #5c7189 !important;
    font-size: 16px;
    line-height: 1.65;
    margin: 0;
}

/* ---------- Section headings ---------- */
.section-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 30px 0 13px;
    animation: revealUp .65s ease both;
}

.section-icon {
    width: 40px;
    height: 40px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: rgba(37,99,235,.10);
    border: 1px solid rgba(37,99,235,.14);
    font-size: 19px;
    transition: transform .25s ease, box-shadow .25s ease;
}

.section-icon:hover {
    transform: translateY(-3px) rotate(-3deg);
    box-shadow: 0 10px 22px rgba(37,99,235,.14);
}

.section-title {
    color: #102a56 !important;
    font-size: 25px;
    font-weight: 820;
    letter-spacing: -.5px;
}

.section-note {
    color: var(--muted) !important;
    font-size: 14px;
    margin: 0 0 17px 52px;
}

/* ---------- Glass cards ---------- */
.glass-card {
    padding: 23px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.82);
    background: var(--glass);
    box-shadow: 0 12px 35px rgba(15,23,42,.07);
    backdrop-filter: blur(15px);
    transition:
        transform .28s cubic-bezier(.2,.8,.2,1),
        box-shadow .28s ease,
        border-color .28s ease;
    animation: revealUp .7s ease both;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 22px 52px rgba(15,23,42,.12);
    border-color: rgba(96,165,250,.35);
}

.card-kicker {
    color: #52739b !important;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.card-title {
    color: #12345c !important;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 6px;
}

.card-copy {
    color: #647891 !important;
    font-size: 13px;
    line-height: 1.55;
}

/* ---------- Inputs ---------- */
[data-testid="stWidgetLabel"] p {
    color: #284562 !important;
    font-weight: 750 !important;
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.92) !important;
    border: 1px solid #cfdae7 !important;
    border-radius: 13px !important;
    min-height: 46px !important;
    box-shadow: 0 5px 17px rgba(15,23,42,.035) !important;
    transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}

div[data-baseweb="select"] > div:hover {
    border-color: #8fb7ea !important;
    box-shadow: 0 9px 23px rgba(37,99,235,.08) !important;
    transform: translateY(-1px);
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #1e334a !important;
}

[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,.92) !important;
    color: #1e334a !important;
    border: none !important;
    font-weight: 600 !important;
}

[data-testid="stNumberInput"] > div {
    background: rgba(255,255,255,.92) !important;
    border: 1px solid #cfdae7 !important;
    border-radius: 13px !important;
    transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}

[data-testid="stNumberInput"] > div:focus-within {
    border-color: #7aa9e8 !important;
    box-shadow: 0 9px 24px rgba(37,99,235,.10) !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    border: 1px dashed #9bb6d5 !important;
    border-radius: 20px !important;
    padding: 17px !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,.86), rgba(243,248,255,.75)) !important;
    box-shadow: 0 12px 32px rgba(15,23,42,.06) !important;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

[data-testid="stFileUploader"]:hover {
    transform: translateY(-3px);
    border-color: #4d8de0 !important;
    box-shadow: 0 19px 42px rgba(37,99,235,.11) !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

[data-testid="stFileUploader"] span {
    color: #334d69 !important;
}

[data-testid="stFileUploader"] small {
    color: #71849a !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    position: relative;
    overflow: hidden;
    width: 100%;
    min-height: 58px;
    border: none !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #15386f 0%, #2563eb 52%, #6255ee 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: .01em;
    box-shadow: 0 14px 32px rgba(37,99,235,.25) !important;
    transition: transform .16s ease, box-shadow .2s ease, filter .2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.005) !important;
    box-shadow: 0 20px 42px rgba(37,99,235,.31) !important;
    filter: saturate(1.08);
}

.stButton > button:active {
    transform: translateY(1px) scale(.985) !important;
    box-shadow: 0 7px 17px rgba(37,99,235,.22) !important;
}

.stButton > button:after {
    content: "";
    position: absolute;
    top: 0;
    left: -90%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.23), transparent);
    transform: skewX(-18deg);
}

.stButton > button:hover:after {
    animation: sweep .8s ease;
}

.stButton > button p {
    color: white !important;
}

/* ---------- Image ---------- */
[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 1px solid rgba(255,255,255,.85);
    box-shadow: 0 16px 40px rgba(15,23,42,.12) !important;
    transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.012);
    box-shadow: 0 23px 52px rgba(15,23,42,.16) !important;
}

/* ---------- Result cards ---------- */
.result-positive,
.result-negative {
    position: relative;
    overflow: hidden;
    padding: 28px;
    border-radius: 24px;
    margin-top: 12px;
    animation: resultReveal .75s cubic-bezier(.2,.8,.2,1) both;
}

.result-positive {
    background: linear-gradient(135deg, rgba(255,245,245,.94), rgba(255,235,239,.78));
    border: 1px solid rgba(225,29,72,.16);
}

.result-negative {
    background: linear-gradient(135deg, rgba(240,253,247,.95), rgba(232,249,242,.80));
    border: 1px solid rgba(16,185,129,.18);
}

.result-positive:after,
.result-negative:after {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    right: -70px;
    top: -95px;
    border-radius: 50%;
    background: rgba(255,255,255,.30);
    animation: softFloat 5s ease-in-out infinite;
}

.result-label {
    font-size: 11px;
    font-weight: 850;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: #60748b !important;
}

.result-title {
    position: relative;
    z-index: 1;
    margin: 6px 0 8px;
    color: #102a56 !important;
    font-size: 28px;
    line-height: 1.15;
    font-weight: 850;
}

.result-copy {
    position: relative;
    z-index: 1;
    color: #536a82 !important;
    line-height: 1.65;
    font-size: 14px;
    max-width: 850px;
}

/* ---------- Status pill ---------- */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(148,163,184,.22);
    color: #3d5874 !important;
    font-size: 12px;
    font-weight: 800;
    margin-top: 11px;
}

.pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 0 rgba(34,197,94,.42);
    animation: pulse 1.9s infinite;
}

/* ---------- Progress / analysis animation ---------- */
.analysis-shell {
    padding: 23px;
    margin-top: 18px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.8);
    background: rgba(255,255,255,.68);
    box-shadow: 0 13px 36px rgba(15,23,42,.07);
    animation: revealUp .5s ease both;
}

.analysis-line {
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: #e5edf7;
    margin-top: 13px;
}

.analysis-line > div {
    width: 42%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #2563eb, #6d5dfc, #22d3ee);
    animation: scan 1.35s ease-in-out infinite;
}

/* ---------- Small information boxes ---------- */
.info-box {
    padding: 18px 19px;
    border-radius: 18px;
    background: rgba(247,250,255,.80);
    border: 1px solid #dce6f1;
    color: #5c7189 !important;
    line-height: 1.62;
    font-size: 13px;
    transition: transform .25s ease, box-shadow .25s ease;
}

.info-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 13px 30px rgba(15,23,42,.07);
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.70) !important;
    border: 1px solid #dce6f1 !important;
    border-radius: 17px !important;
    box-shadow: 0 8px 25px rgba(15,23,42,.04);
}

/* ---------- Divider ---------- */
hr {
    border: none !important;
    border-top: 1px solid rgba(148,163,184,.24) !important;
    margin: 27px 0 !important;
}

/* ---------- Animations ---------- */
@keyframes revealUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes resultReveal {
    from { opacity: 0; transform: translateY(20px) scale(.985); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes softFloat {
    0%, 100% { transform: translate3d(0,0,0); }
    50% { transform: translate3d(0,-7px,0); }
}

@keyframes drift {
    0%, 100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(-12px,10px) scale(1.05); }
}

@keyframes sweep {
    from { left: -90%; }
    to { left: 130%; }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,.38); }
    70% { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

@keyframes scan {
    0% { transform: translateX(-110%); }
    100% { transform: translateX(260%); }
}

@media (prefers-reduced-motion: reduce) {
    *, *:before, *:after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}
</style>
""",
unsafe_allow_html=True,
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLINICAL_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "batten_disease_best_model.pkl"
)

MRI_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "batten_mri_efficientnet.keras"
)

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "processed", "batten_combined_dataset.csv"
)

# ============================================================
# MODEL / DATA LOADERS
# ============================================================

@st.cache_resource
def load_clinical_model():
    return joblib.load(CLINICAL_MODEL_PATH)


@st.cache_data
def load_clinical_values():
    df = pd.read_csv(DATASET_PATH)

    phenotype_values = sorted(
        df["Phenotype"].dropna().astype(str).unique().tolist()
    )

    histology_values = sorted(
        df["Histology"].dropna().astype(str).unique().tolist()
    )

    country_values = sorted(
        df["Country_of_origin"].dropna().astype(str).unique().tolist()
    )

    return phenotype_values, histology_values, country_values


@st.cache_resource
def load_mri_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MRI_MODEL_PATH)


# ============================================================
# INITIALIZE CLINICAL SYSTEM
# ============================================================

try:
    clinical_model = load_clinical_model()
    phenotype_values, histology_values, country_values = load_clinical_values()
except Exception as e:
    st.error("Unable to initialize the patient information system.")
    st.error(str(e))
    st.stop()


# ============================================================
# MRI PREPROCESSING
# ============================================================

def preprocess_mri(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# MRI PREDICTION
# ============================================================

def predict_mri(model, image):
    processed_image = preprocess_mri(image)

    raw_probability = float(
        model.predict(processed_image, verbose=0)[0][0]
    )

    # Training folders:
    # batten = 0
    # non_batten = 1
    # Sigmoid output = P(non_batten)
    non_batten_probability = raw_probability
    batten_probability = 1.0 - non_batten_probability

    return (
        batten_probability,
        non_batten_probability,
        processed_image,
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, image_array, batten_class=True):
    try:
        import tensorflow as tf

        backbone = model.get_layer("efficientnetb0")
        target_layer = None

        for layer in reversed(backbone.layers):
            try:
                output_shape = layer.output.shape
                if len(output_shape) == 4:
                    target_layer = layer
                    break
            except Exception:
                continue

        if target_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=[target_layer.output, backbone.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, backbone_output = grad_model(
                image_array, training=False
            )

            x = model.get_layer("global_average_pooling2d")(
                backbone_output
            )

            x = model.get_layer("dropout")(
                x, training=False
            )

            predictions = model.get_layer("dense")(x)

            if batten_class:
                target = 1.0 - predictions[:, 0]
            else:
                target = predictions[:, 0]

        gradients = tape.gradient(target, conv_outputs)

        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(
            gradients, axis=(1, 2)
        )

        conv_outputs = conv_outputs[0]
        pooled_gradients = pooled_gradients[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_gradients,
            axis=-1,
        )

        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)

        if float(maximum) > 0:
            heatmap /= maximum

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (image_array.shape[2], image_array.shape[1]),
        )

        original = image_array[0].astype(np.uint8)

        heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB,
        )

        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap_color,
            0.40,
            0,
        )

        return overlay

    except Exception:
        return None


# ============================================================
# HEADER
# ============================================================

st.markdown(
"""
<div class="hero">
    <div class="brand-row">
        <div class="brand-mark">🧬</div>
        <div class="brand-name">Batten-AI</div>
    </div>
    <h1>Batten Disease Prediction</h1>
    <p class="hero-sub">
        An interactive AI-assisted prediction interface combining
        patient clinical context with brain MRI analysis and
        explainable visual attention.
    </p>
    <div class="status-pill">
        <span class="pulse"></span>
        Analysis system ready
    </div>
</div>
""",
unsafe_allow_html=True,
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
"""
<div class="section-head">
    <div class="section-icon">👤</div>
    <div class="section-title">Patient Information</div>
</div>
<p class="section-note">
    Enter the clinical information associated with the MRI being assessed.
</p>
""",
unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        '<div class="card-kicker">Patient profile</div>'
        '<div class="card-title">Basic clinical context</div>',
        unsafe_allow_html=True,
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other / Not specified"],
        help=(
            "Recorded as patient context. Gender is not currently "
            "a feature of the trained clinical model because the "
            "source training tables do not contain reliable gender labels."
        ),
    )

    phenotype = st.selectbox(
        "Clinical presentation",
        phenotype_values,
        help=(
            "Choose the clinical presentation that most closely "
            "matches the patient's record."
        ),
    )

    age_at_onset = st.number_input(
        "Age at onset (years)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        help="Enter the age when the first relevant symptoms were observed.",
    )

with col2:
    st.markdown(
        '<div class="card-kicker">Clinical record</div>'
        '<div class="card-title">Additional context</div>',
        unsafe_allow_html=True,
    )

    histology = st.selectbox(
        "Histology",
        histology_values,
        help="Select the histological finding recorded for the patient.",
    )

    country = st.selectbox(
        "Country of origin",
        country_values,
        help="Select the country or geographical description recorded for the patient.",
    )

    st.markdown(
        """
        <div class="info-box">
            <strong>Why this information matters</strong><br>
            Clinical context provides additional patient information for
            the assessment workflow. The current clinical model predicts
            NCL subtype/context rather than a binary Batten-vs-normal
            diagnosis, so it is not converted into a fabricated disease
            probability.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MRI
# ============================================================

st.markdown(
"""
<div class="section-head">
    <div class="section-icon">🧠</div>
    <div class="section-title">Brain MRI</div>
</div>
<p class="section-note">
    Upload the brain MRI corresponding to this patient.
</p>
""",
unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image in JPG, JPEG or PNG format.",
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="card-kicker">MRI preview</div>'
        '<div class="card-title">Uploaded scan</div>',
        unsafe_allow_html=True,
    )

    st.image(
        image,
        caption="Patient MRI",
        use_container_width=True,
    )


# ============================================================
# ANALYZE
# ============================================================

st.markdown(
"""
<div class="section-head">
    <div class="section-icon">✦</div>
    <div class="section-title">AI Assessment</div>
</div>
<p class="section-note">
    Start the analysis after entering the patient information and uploading an MRI.
</p>
""",
unsafe_allow_html=True,
)

analyze = st.button(
    "Analyze Patient",
    use_container_width=True,
    type="primary",
)

# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if image is None:
        st.warning(
            "Please upload the patient's brain MRI before starting the analysis."
        )
        st.stop()

    # --------------------------------------------------------
    # Patient clinical input
    # --------------------------------------------------------

    clinical_input = pd.DataFrame(
        [{
            "Phenotype": phenotype,
            "Age_at_onset": age_at_onset,
            "Histology": histology,
            "Country_of_origin": country,
        }]
    )

    expected_features = list(clinical_model.feature_names_in_)

    clinical_input = clinical_input[expected_features]

    # --------------------------------------------------------
    # Clinical analysis
    # --------------------------------------------------------

    with st.status("Reviewing patient information...", expanded=False) as status:
        try:
            clinical_prediction = str(
                clinical_model.predict(clinical_input)[0]
            )

            clinical_probabilities = clinical_model.predict_proba(
                clinical_input
            )[0]

            clinical_confidence = float(
                np.max(clinical_probabilities)
            )

            status.update(
                label="Patient information reviewed",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(
                label="Patient information review failed",
                state="error",
                expanded=False,
            )
            st.error("Clinical analysis could not be completed.")
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # MRI analysis
    # --------------------------------------------------------

    with st.status("Analyzing brain MRI...", expanded=True) as status:
        st.markdown(
            """
            <div class="analysis-shell">
                <div class="card-kicker">Deep image analysis</div>
                <div class="card-title">Examining the uploaded MRI</div>
                <div class="card-copy">
                    The system is processing the image and evaluating
                    visual patterns represented in the research training data.
                </div>
                <div class="analysis-line"><div></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            mri_model = load_mri_model()

            (
                batten_probability,
                non_batten_probability,
                processed_image,
            ) = predict_mri(mri_model, image)

            status.update(
                label="MRI analysis completed",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(
                label="MRI analysis failed",
                state="error",
                expanded=False,
            )
            st.error("MRI analysis could not be completed.")
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    with st.status(
        "Generating visual explanation...",
        expanded=False,
    ) as status:

        gradcam_image = generate_gradcam(
            mri_model,
            processed_image,
            batten_class=(
                batten_probability >= non_batten_probability
            ),
        )

        status.update(
            label="Visual explanation generated"
            if gradcam_image is not None
            else "Visual explanation unavailable",
            state="complete" if gradcam_image is not None else "error",
            expanded=False,
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    batten_detected = batten_probability >= 0.50
    st.session_state["batten_detected"] = batten_detected

    # Store only presentation-safe values in session state.
    st.session_state["assessment_complete"] = True

    # ========================================================
    # FINAL ASSESSMENT
    # ========================================================

    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon">✓</div>
            <div class="section-title">Final Patient Assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if batten_detected:
        st.markdown(
            """
            <div class="result-positive">
                <div class="result-label">Assessment result</div>
                <div class="result-title">Batten Disease Pattern Detected</div>
                <div class="result-copy">
                    The MRI analysis identified visual patterns that were
                    more consistent with the Batten disease patterns
                    represented in the research training data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-negative">
                <div class="result-label">Assessment result</div>
                <div class="result-title">Batten Disease Pattern Not Detected</div>
                <div class="result-copy">
                    The MRI analysis did not identify a sufficiently strong
                    visual pattern corresponding to the Batten disease
                    patterns represented in the research training data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-box" style="margin-top:14px;">
                <strong>Important:</strong> A negative Batten assessment
                does not identify another disease or medical condition.
                The current MRI system is specifically designed for
                Batten-vs-non-Batten pattern assessment and cannot provide
                an alternative diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon">▦</div>
            <div class="section-title">Patient Summary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2 = st.columns(2, gap="large")

    with summary_col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-kicker">Patient profile</div>
                <div class="card-title">Recorded information</div>
                <div class="info-box">
                    <strong>Gender:</strong> {gender}<br><br>
                    <strong>Clinical presentation:</strong> {phenotype}<br><br>
                    <strong>Age at onset:</strong> {age_at_onset:g} years
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-kicker">Clinical record</div>
                <div class="card-title">Additional information</div>
                <div class="info-box">
                    <strong>Histology:</strong> {histology}<br><br>
                    <strong>Country of origin:</strong> {country}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

   # ========================================================
    # ========================================================
    # WHY DID THE AI MAKE THIS ASSESSMENT?
    # ========================================================

    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon">?</div>
            <div class="section-title">Why did the AI make this assessment?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if batten_detected:
        explanation = f"""
        <strong>MRI evidence:</strong><br>
        The MRI model identified a visual pattern that was more consistent
        with the Batten-associated patterns represented in its research
        training data. The model's Batten-pattern score was
        <strong>{batten_probability:.1%}</strong>.<br><br>

        <strong>Clinical context:</strong><br>
        The clinical model's predicted NCL subtype was
        <strong>{clinical_prediction}</strong>, with a subtype-classification
        confidence of <strong>{clinical_confidence:.1%}</strong>.<br><br>

        <strong>How these results are used:</strong><br>
        The MRI result provides the primary Batten-vs-non-Batten image
        assessment, while the clinical model provides NCL subtype context.
        The two outputs are kept separate because the system has not been
        trained as a calibrated multimodal diagnostic model.
        """
    else:
        explanation = f"""
        <strong>MRI evidence:</strong><br>
        The MRI model did not identify a sufficiently strong visual pattern
        corresponding to the Batten-associated patterns represented in its
        research training data. The estimated Batten-pattern score was
        <strong>{batten_probability:.1%}</strong>.<br><br>

        <strong>Clinical context:</strong><br>
        The clinical model's predicted NCL subtype was
        <strong>{clinical_prediction}</strong>, with a subtype-classification
        confidence of <strong>{clinical_confidence:.1%}</strong>.<br><br>

        <strong>How to interpret this result:</strong><br>
        This is a negative assessment from the current MRI model, not a
        diagnosis of another neurological condition. If clinical symptoms
        remain concerning, professional neurological and radiological
        evaluation should continue.
        """

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-kicker">Assessment reasoning</div>
            <div class="card-title">Image evidence and clinical context</div>
            <div class="card-copy">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # MRI ATTENTION
    # ========================================================

    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon">◉</div>
            <div class="section-title">MRI Analysis & AI Attention Map</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if gradcam_image is not None:
        st.image(
            gradcam_image,
            caption=(
                "Highlighted regions indicate areas that contributed "
                "more strongly to the AI's visual assessment."
            ),
            use_container_width=True,
        )

        attention_html = """
<div class="info-box" style="margin-top:12px;">
<strong>How to interpret the AI attention map</strong><br><br>

The highlighted regions represent areas that contributed more
strongly to the neural network's MRI classification. The model
learned these visual patterns from the MRI images used during
research training.<br><br>

<strong>Medical context:</strong><br>
Neuroimaging studies of neuronal ceroid lipofuscinoses have
reported abnormalities such as cerebral or cerebellar volume loss,
thalamic signal changes and white-matter abnormalities. The
specific pattern can vary according to NCL subtype and disease
stage.<br><br>

<strong>Important limitation:</strong><br>
The current Grad-CAM does not perform anatomical segmentation.
Therefore, the highlighted area should not automatically be
labelled as the thalamus, cerebellum, cortex or another specific
anatomical structure. It represents model evidence rather than a
confirmed lesion.<br><br>

<strong>Clinical interpretation:</strong><br>
A radiologist or neurologist must determine whether the highlighted
region corresponds to a genuine structural or signal abnormality
and whether that abnormality is compatible with the patient's
clinical presentation.
</div>
"""

        st.markdown(
            attention_html,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "The AI attention visualization could not be generated for this MRI."
        )

    # ========================================================
    # NEXT STEPS
    # ========================================================

    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon">→</div>
            <div class="section-title">Recommended Next Steps</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if batten_detected:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-kicker">Clinical follow-up</div>
                <div class="card-title">Consider professional evaluation</div>
                <div class="card-copy">
                    A positive research-prototype assessment should be
                    followed by appropriate clinical evaluation.
                </div>
                <div class="info-box" style="margin-top:14px;">
                    • Neurological assessment.<br>
                    • Ophthalmological assessment where appropriate.<br>
                    • Genetic evaluation and molecular testing.<br>
                    • Evaluation of seizures, movement, developmental
                      changes and vision.<br>
                    • Swallowing, communication, mobility and nutritional
                      assessment when clinically indicated.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-kicker">Clinical follow-up</div>
                <div class="card-title">Continue evaluation when symptoms persist</div>
                <div class="card-copy">
                    A negative result means that this system did not
                    identify a strong Batten disease pattern in the
                    uploaded MRI. If neurological or other symptoms
                    continue, professional medical evaluation remains
                    appropriate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# NUTRITION & DAILY CARE
# ============================================================

batten_detected = st.session_state.get("batten_detected", False)

st.markdown(
    """
<div class="section-head">
    <div class="section-icon">+</div>
    <div class="section-title">Nutrition & Daily Care</div>
</div>
""",
    unsafe_allow_html=True,
)

if batten_detected:

    st.markdown(
        """
<div class="info-box">
<strong>Supportive care when a Batten-associated MRI pattern is detected</strong><br><br>

<strong>1. Nutrition and hydration</strong><br>
Maintain adequate calorie, fluid and nutrient intake. Nutritional
status and growth should be monitored as part of ongoing clinical care.<br><br>

<strong>2. Swallowing and feeding</strong><br>
If there is difficulty swallowing, choking, coughing during meals,
weight loss or recurrent respiratory problems, a clinical swallowing
and feeding assessment should be considered.<br><br>

<strong>3. Food consistency</strong><br>
When swallowing difficulty is present, food and liquid consistency
should be individualized by the appropriate clinical team to reduce
aspiration risk.<br><br>

<strong>4. When oral intake becomes insufficient</strong><br>
If nutritional requirements cannot be met safely by mouth, or
aspiration risk becomes significant, the treating team may discuss
enteral feeding options with the family.<br><br>

<strong>5. Seizure-specific diets</strong><br>
There is no universal Batten disease diet. A specialized dietary
therapy for seizure control should only be considered under the
supervision of the treating neurologist and an appropriately
qualified dietitian.
</div>
""",
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
<div class="info-box">
<strong>General nutrition and well-being</strong><br><br>

The current MRI assessment did not identify a strong Batten-associated
pattern. Therefore, Batten-specific dietary recommendations are not
generated from this result.<br><br>

Maintain adequate hydration and balanced nutrition appropriate to
the individual's age and nutritional requirements.<br><br>

If neurological symptoms, swallowing problems, unexplained weight
loss or other concerning symptoms persist, appropriate clinical
evaluation should still be continued.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.markdown(
    """
<div class="info-box">
    <strong>Research prototype</strong><br>
    The AI assessment does not constitute a
    medical diagnosis and should not replace evaluation by
    qualified healthcare professionals.
</div>
""",
    unsafe_allow_html=True,
)


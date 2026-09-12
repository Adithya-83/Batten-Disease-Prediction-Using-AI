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
    page_title="Batten Disease AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PREMIUM DASHBOARD UI
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL PAGE
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(59, 130, 246, 0.16),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 8%,
            rgba(139, 92, 246, 0.13),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(14, 165, 233, 0.10),
            transparent 30%
        ),
        #eef3f9 !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ==========================================================
   ALL NORMAL TEXT
   ========================================================== */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: #24364d;
}


/* ==========================================================
   MAIN TITLE
   ========================================================== */

h1 {
    color: #102a56 !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin-bottom: 4px !important;
}

h2 {
    color: #102a56 !important;
    font-size: 27px !important;
    font-weight: 750 !important;
}

h3 {
    color: #183b68 !important;
    font-weight: 700 !important;
}


/* ==========================================================
   CAPTION
   ========================================================== */

[data-testid="stCaptionContainer"] {
    color: #5d718b !important;
}

[data-testid="stCaptionContainer"] p {
    color: #5d718b !important;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border: none !important;
    border-top: 1px solid #d5dfeb !important;
    margin-top: 30px !important;
    margin-bottom: 30px !important;
}


/* ==========================================================
   SELECT BOX
   ========================================================== */

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd8e7 !important;
    border-radius: 11px !important;
    min-height: 45px !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04) !important;
}

div[data-baseweb="select"] span {
    color: #1e293b !important;
}

div[data-baseweb="select"] input {
    color: #1e293b !important;
}


/* ==========================================================
   NUMBER INPUT
   ========================================================== */

[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: none !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInput"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd8e7 !important;
    border-radius: 11px !important;
}


/* ==========================================================
   INPUT LABELS
   ========================================================== */

[data-testid="stWidgetLabel"] p {
    color: #243b5a !important;
    font-weight: 700 !important;
}


/* ==========================================================
   HELP ICON
   ========================================================== */

[data-testid="stWidgetLabel"] svg {
    color: #52739b !important;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid #cbd8e7 !important;
    border-radius: 15px !important;
    padding: 18px !important;
    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.06) !important;
}

[data-testid="stFileUploader"] section {
    background: #ffffff !important;
}

[data-testid="stFileUploader"] span {
    color: #334155 !important;
}

[data-testid="stFileUploader"] small {
    color: #64748b !important;
}


/* ==========================================================
   PRIMARY BUTTON
   ========================================================== */

.stButton > button {
    width: 100%;
    min-height: 58px;

    border-radius: 14px !important;
    border: none !important;

    background:
        linear-gradient(
            135deg,
            #173b7a 0%,
            #2563eb 55%,
            #4f46e5 100%
        ) !important;

    color: #ffffff !important;

    font-size: 17px !important;
    font-weight: 750 !important;

    box-shadow:
        0 10px 24px rgba(37, 99, 235, 0.25) !important;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;

    box-shadow:
        0 14px 30px rgba(37, 99, 235, 0.32) !important;
}

.stButton > button p {
    color: #ffffff !important;
}


/* ==========================================================
   METRIC
   ========================================================== */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f5f8fc
        ) !important;

    border: 1px solid #d5dfeb !important;

    border-radius: 15px !important;

    padding: 20px !important;

    box-shadow:
        0 8px 24px rgba(15, 23, 42, 0.06) !important;
}

[data-testid="stMetricLabel"] {
    color: #526781 !important;
}

[data-testid="stMetricValue"] {
    color: #173b7a !important;
}


/* ==========================================================
   ALERTS
   ========================================================== */

[data-testid="stAlert"] {
    border-radius: 14px !important;
}


/* ==========================================================
   IMAGE
   ========================================================== */

[data-testid="stImage"] img {
    border-radius: 15px !important;
    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.10) !important;
}


/* ==========================================================
   EXPANDERS
   ========================================================== */

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d8e2ee !important;
    border-radius: 13px !important;
}


/* ==========================================================
   FOOTER
   ========================================================== */

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CLINICAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "batten_disease_best_model.pkl"
)

MRI_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "batten_mri_efficientnet.keras"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "batten_combined_dataset.csv"
)


# ============================================================
# LOAD CLINICAL MODEL
# ============================================================

@st.cache_resource
def load_clinical_model():

    return joblib.load(
        CLINICAL_MODEL_PATH
    )


# ============================================================
# LOAD CLINICAL VALUES
# ============================================================

@st.cache_data
def load_clinical_values():

    df = pd.read_csv(
        DATASET_PATH
    )

    phenotype_values = sorted(
        df["Phenotype"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    histology_values = sorted(
        df["Histology"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    country_values = sorted(
        df["Country_of_origin"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return (
        phenotype_values,
        histology_values,
        country_values
    )


# ============================================================
# INITIALIZE CLINICAL SYSTEM
# ============================================================

try:

    clinical_model = load_clinical_model()

    (
        phenotype_values,
        histology_values,
        country_values
    ) = load_clinical_values()

except Exception as e:

    st.error(
        "Unable to initialize the patient information system."
    )

    st.error(
        str(e)
    )

    st.stop()


# ============================================================
# LAZY MRI MODEL
# ============================================================

@st.cache_resource
def load_mri_model():

    import tensorflow as tf

    return tf.keras.models.load_model(
        MRI_MODEL_PATH
    )


# ============================================================
# MRI PREPROCESSING
# ============================================================

def preprocess_mri(
    image
):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# MRI PREDICTION
# ============================================================

def predict_mri(
    model,
    image
):

    processed_image = preprocess_mri(
        image
    )

    raw_probability = float(
        model.predict(
            processed_image,
            verbose=0
        )[0][0]
    )

    # Training folders:
    #
    # batten     = 0
    # non_batten = 1
    #
    # Sigmoid output = P(non_batten)

    non_batten_probability = (
        raw_probability
    )

    batten_probability = (
        1.0 -
        non_batten_probability
    )

    return (
        batten_probability,
        non_batten_probability,
        processed_image
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_array,
    batten_class=True
):

    try:

        import tensorflow as tf

        backbone = model.get_layer(
            "efficientnetb0"
        )

        target_layer = None

        for layer in reversed(
            backbone.layers
        ):

            try:

                output_shape = (
                    layer.output.shape
                )

                if len(output_shape) == 4:

                    target_layer = layer

                    break

            except Exception:

                continue

        if target_layer is None:

            return None

        grad_model = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=[
                target_layer.output,
                backbone.output
            ]
        )

        with tf.GradientTape() as tape:

            (
                conv_outputs,
                backbone_output
            ) = grad_model(
                image_array,
                training=False
            )

            x = model.get_layer(
                "global_average_pooling2d"
            )(backbone_output)

            x = model.get_layer(
                "dropout"
            )(
                x,
                training=False
            )

            predictions = model.get_layer(
                "dense"
            )(x)

            if batten_class:

                target = (
                    1.0 -
                    predictions[:, 0]
                )

            else:

                target = (
                    predictions[:, 0]
                )

        gradients = tape.gradient(
            target,
            conv_outputs
        )

        if gradients is None:

            return None

        pooled_gradients = (
            tf.reduce_mean(
                gradients,
                axis=(1, 2)
            )
        )

        conv_outputs = conv_outputs[0]

        pooled_gradients = (
            pooled_gradients[0]
        )

        heatmap = tf.reduce_sum(
            conv_outputs *
            pooled_gradients,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        maximum = tf.reduce_max(
            heatmap
        )

        if float(maximum) > 0:

            heatmap /= maximum

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (
                image_array.shape[2],
                image_array.shape[1]
            )
        )

        original = (
            image_array[0]
            .astype(np.uint8)
        )

        heatmap_uint8 = np.uint8(
            255 *
            heatmap
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        return overlay

    except Exception:

        return None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧬 Batten Disease Prediction Using Artificial Intelligence"
)

st.caption(
    "AI-assisted multimodal assessment using patient information and brain MRI"
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.divider()

st.header(
    "👤 Patient Information"
)

st.write(
    "Enter the clinical information corresponding to the uploaded MRI."
)


col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other / Not specified"
        ],
        help=(
            "Select the patient's gender. This information "
            "is recorded as patient context and does not alter "
            "the currently trained clinical model."
        )
    )


    phenotype = st.selectbox(
        "Phenotype",
        phenotype_values,
        help=(
            "Select the clinical presentation that best "
            "matches the patient's record."
        )
    )


    age_at_onset = st.number_input(
        "Age at onset",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        help=(
            "Enter the patient's age when symptoms were first observed."
        )
    )


with col2:

    histology = st.selectbox(
        "Histology",
        histology_values,
        help=(
            "Select the histological finding corresponding "
            "to the patient's clinical record."
        )
    )


    country = st.selectbox(
        "Country of origin",
        country_values,
        help=(
            "Select the country or geographical description "
            "recorded for the patient."
        )
    )


# ============================================================
# MRI
# ============================================================

st.divider()

st.header(
    "🧠 Brain MRI"
)

st.write(
    "Upload the brain MRI corresponding to this patient's clinical information."
)


uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload a brain MRI image in JPG, JPEG or PNG format."
    )
)


image = None


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Patient MRI",
        use_container_width=True
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Patient",
    use_container_width=True,
    type="primary"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # MRI validation
    # --------------------------------------------------------

    if image is None:

        st.warning(
            "Please upload the patient's brain MRI before starting the analysis."
        )

        st.stop()


    # ========================================================
    # CLINICAL INPUT
    # ========================================================

    clinical_input = pd.DataFrame(
        [{
            "Phenotype": phenotype,
            "Age_at_onset": age_at_onset,
            "Histology": histology,
            "Country_of_origin": country
        }]
    )


    expected_features = list(
        clinical_model.feature_names_in_
    )


    clinical_input = clinical_input[
        expected_features
    ]


    # ========================================================
    # CLINICAL ANALYSIS
    # ========================================================

    try:

        clinical_prediction = str(
            clinical_model.predict(
                clinical_input
            )[0]
        )

        clinical_probabilities = (
            clinical_model.predict_proba(
                clinical_input
            )[0]
        )

        clinical_confidence = float(
            np.max(
                clinical_probabilities
            )
        )

    except Exception as e:

        st.error(
            "Clinical analysis could not be completed."
        )

        st.error(
            str(e)
        )

        st.stop()


    # ========================================================
    # MRI ANALYSIS
    # ========================================================

    with st.spinner(
        "Performing deep MRI analysis..."
    ):

        try:

            mri_model = load_mri_model()

            (
                batten_probability,
                non_batten_probability,
                processed_image
            ) = predict_mri(
                mri_model,
                image
            )

        except Exception as e:

            st.error(
                "MRI analysis could not be completed."
            )

            st.error(
                str(e)
            )

            st.stop()


    # ========================================================
    # GRAD-CAM
    # ========================================================

    with st.spinner(
        "Examining MRI regions that influenced the assessment..."
    ):

        gradcam_image = generate_gradcam(
            mri_model,
            processed_image,
            batten_class=(
                batten_probability >=
                non_batten_probability
            )
        )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    batten_detected = (
        batten_probability >= 0.50
    )


    # ========================================================
    # FINAL PATIENT ASSESSMENT
    # ========================================================

    st.divider()

    st.header(
        "📋 Final Patient Assessment"
    )


    if batten_detected:

        st.error(
            "⚠️ Batten Disease Pattern Detected"
        )

        st.write(
            """
            The AI analysis identified visual patterns in
            the uploaded brain MRI that are more consistent
            with Batten disease patterns represented in the
            training data.
            """
        )

    else:

        st.success(
            "✅ Batten Disease Pattern Not Detected"
        )

        st.write(
            """
            The AI analysis did not identify a visual pattern
            strongly consistent with Batten disease in the
            uploaded MRI.
            """
        )

        st.info(
            """
            This result does not identify another disease
            or medical condition. The current system is
            specifically designed to assess Batten disease
            patterns and cannot provide an alternative
            diagnosis.
            """
        )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.divider()

    st.header(
        "🧾 Patient Summary"
    )


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Phenotype:** {phenotype}"
        )

        st.write(
            f"**Age at onset:** {age_at_onset:g} years"
        )


    with summary_col2:

        st.write(
            f"**Histology:** {histology}"
        )

        st.write(
            f"**Country of origin:** {country}"
        )


    # ========================================================
    # WHY?
    # ========================================================

    st.divider()

    st.header(
        "🔎 Why did the AI make this assessment?"
    )


    if batten_detected:

        st.write(
            """
            The system analyzed the uploaded brain MRI
            together with the clinical information supplied
            for the same patient.

            The MRI component identified visual patterns
            that contributed to a Batten disease assessment.

            The highlighted regions below show where the
            neural network placed greater importance while
            producing the MRI prediction.
            """
        )

    else:

        st.write(
            """
            The system analyzed the uploaded brain MRI
            together with the clinical information supplied
            for the same patient.

            The MRI component did not identify a sufficiently
            strong visual pattern corresponding to the Batten
            disease patterns represented in the training data.
            """
        )


    # ========================================================
    # MRI ATTENTION MAP
    # ========================================================

    st.header(
        "🧠 MRI Analysis & AI Attention Map"
    )


    if gradcam_image is not None:

        st.image(
            gradcam_image,
            caption=(
                "Highlighted regions indicate areas that "
                "contributed more strongly to the AI's "
                "visual assessment."
            ),
            use_container_width=True
        )


        st.write(
            """
            **How to understand the highlighted regions**

            The heatmap shows areas of the MRI that had
            greater influence on the neural network's decision.

            These regions represent the AI's evidence areas.
            They should not be interpreted as confirmed lesions
            or as proof that a specific anatomical region is
            the cause of Batten disease.
            """
        )


        st.warning(
            """
            Grad-CAM is an AI explainability method. It shows
            where the model focused when making its prediction;
            it does not establish a confirmed medical abnormality
            or replace professional MRI interpretation.
            """
        )


    else:

        st.info(
            "The AI attention visualization could not be generated for this MRI."
        )


    # ========================================================
    # NEXT STEPS
    # ========================================================

    st.divider()

    st.header(
        "🩺 Recommended Next Steps"
    )


    if batten_detected:

        st.write(
            """
            A positive AI assessment should be followed by
            professional medical evaluation.

            Appropriate follow-up may include:

            • Neurological assessment.

            • Ophthalmological assessment where appropriate.

            • Genetic evaluation and molecular testing.

            • Evaluation of seizures, movement, developmental
              changes and vision.

            • Assessment of swallowing, communication, mobility
              and nutritional status when clinically indicated.
            """
        )

    else:

        st.write(
            """
            A negative AI assessment means that the current
            system did not identify a strong Batten disease
            pattern in the uploaded MRI.

            If the patient continues to experience neurological
            or other symptoms, professional medical evaluation
            is still appropriate because this system does not
            diagnose conditions other than Batten disease.
            """
        )


    # ========================================================
    # NUTRITION
    # ========================================================

    st.header(
        "🍎 Nutrition & Daily Care"
    )


    st.info(
        """
        There is no single universal Batten disease diet.

        Nutrition and hydration should be individualized
        according to the patient's age, nutritional status,
        swallowing ability and medical requirements.

        Medical assessment is especially important when there
        is difficulty swallowing, choking, recurrent respiratory
        problems, poor weight gain or difficulty meeting
        nutritional requirements.

        Any specialized dietary approach intended to assist
        seizure control should only be considered under the
        supervision of an appropriate neurologist and dietitian.
        """
    )


    # ========================================================
    # FINAL DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        """
        Research prototype — intended for academic and research
        demonstration. The AI assessment does not constitute a
        medical diagnosis and should not replace evaluation by
        qualified healthcare professionals.
        """
    )
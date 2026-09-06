import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Batten Disease Prediction Using AI",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "batten_disease_best_model.pkl"
)


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.error(e)

    st.stop()


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🧬 Batten Disease Prediction Using AI")

st.write(
    """
    This AI-based prototype analyzes available patient clinical
    and genetic information
    """
)

# ============================================================
# IMPORTANT INFORMATION
# ============================================================

st.subheader("📋 Patient Information")

st.info(
    """
    Enter the available patient information below.
    Missing information can be left empty where applicable.
    """
)


# ============================================================
# AUTOMATIC FEATURE DETECTION
# ============================================================

try:

    expected_features = list(
        model.feature_names_in_
    )

except:

    expected_features = []


# ============================================================
# USER INPUT
# ============================================================

user_data = {}


if len(expected_features) > 0:

    st.write("Enter Patient Clinical Information")

    columns = st.columns(2)

    for index, feature in enumerate(expected_features):

        with columns[index % 2]:

            feature_lower = feature.lower()

            # Numerical Features
            if any(
                word in feature_lower
                for word in [
                    "age",
                    "onset",
                    "pmid",
                    "position",
                    "score"
                ]
            ):

                value = st.number_input(
                    feature,
                    value=0.0
                )

            # Categorical Features
            else:

                value = st.text_input(
                    feature,
                    value=""
                )

            user_data[feature] = value


else:

    st.warning(
        """
        Automatic feature detection was not available.
        The model will require reconstruction of the
        original training input structure.
        """
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()


if st.button(
    "🔍 Predict Batten Disease",
    use_container_width=True
):

    try:

        if len(expected_features) == 0:

            st.error(
                """
                Model feature names could not be automatically detected.
                Please use the original training dataset feature structure.
                """
            )

            st.stop()


        # Create DataFrame
        input_data = pd.DataFrame(
            [user_data]
        )


        # Ensure correct feature order
        input_data = input_data[
            expected_features
        ]


        # Prediction
        prediction = model.predict(
            input_data
        )[0]


        st.divider()

        st.subheader(
            "🤖 Prediction Result"
        )


        # Display Result
        if prediction == 1:

            st.error(
                "⚠️ Prediction: Possible Batten Disease Pattern Detected"
            )

        else:

            st.success(
                "✅ Prediction: No Batten Disease Pattern Detected"
            )


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        try:

            probability = model.predict_proba(
                input_data
            )

            st.subheader(
                "Prediction Confidence"
            )

            st.write(
                probability
            )

        except:

            st.info(
                "Probability information is not available."
            )


    except Exception as e:

        st.error(
            "Prediction Error"
        )

        st.error(e)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

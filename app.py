
import streamlit as st
import pandas as pd
import joblib


saved_data = joblib.load("heart_model.pkl")
model = saved_data["model"]
features = saved_data["features"]

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the required measurements to get a prediction "
    "from your trained machine learning model."
)

st.warning(
    
    "medical diagnosis or a substitute for professional care."
)


categorical_options = {
    "sex": [0, 1],
    "cp": [0, 1, 2, 3],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [0, 1, 2],
    "ca": [0, 1, 2, 3, 4],
    "thal": [0, 1, 2, 3]
}


numeric_defaults = {
    "age": (45, 1, 120),
    "trestbps": (120, 50, 250),
    "restbps": (120, 50, 250),
    "chol": (200, 50, 700),
    "thalach": (150, 50, 250),
    "oldpeak": (0.0, 0.0, 10.0)
}

input_values = {}

st.subheader("Enter patient measurements")

with st.form("prediction_form"):
    columns = st.columns(2)

    for i, feature in enumerate(features):
        with columns[i % 2]:
            if feature in categorical_options:
                options = categorical_options[feature]
                input_values[feature] = st.selectbox(
                    f"{feature}",
                    options=options,
                    key=f"input_{feature}"
                )

            elif feature in numeric_defaults:
                default, minimum, maximum = numeric_defaults[feature]

                input_values[feature] = st.number_input(
                    feature,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=0.1 if feature == "oldpeak" else 1.0,
                    key=f"input_{feature}"
                )

            else:
                st.error(
                    f"Add an input field for the feature: {feature}"
                )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True
    )

if submitted:
    if all(feature in input_values for feature in features):
        input_df = pd.DataFrame(
            [[input_values[feature] for feature in features]],
            columns=features
        )

        prediction = model.predict(input_df)[0]
        st.write("Prediction:", prediction)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model predicts class 1. "
                "Please do not interpret this result as a diagnosis."
            )
        else:
            st.success(
                "The model predicts class 0. "
                "This does not rule out heart disease."
            )

        st.caption(f"Predicted class: {prediction}")

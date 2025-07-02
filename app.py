import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

st.title("🍷 Wine Quality Predictor")
st.markdown("Upload a CSV or enter wine properties to predict quality.")

# --- Load and Train Model ---
@st.cache_resource
def train_model():
    data = pd.read_csv("WineQT.csv")

    # Remove 'id' column completely
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    if 'quality' not in data.columns:
        st.error("❌ Dataset must have a 'quality' column.")
        st.stop()

    X = data.drop("quality", axis=1)
    y = data["quality"]

    model = make_pipeline(RandomForestClassifier(random_state=42))
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_names = train_model()

# --- Manual Input Section ---
st.subheader("🔢 Enter Wine Features")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

if st.button("🔍 Predict Quality"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Wine Quality: **{prediction}**")

# --- Bulk CSV Upload Section ---
st.subheader("📁 Bulk Prediction")
pred_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if pred_file:
    try:
        pred_df = pd.read_csv(pred_file)

        # Remove 'id' and 'quality' columns if present
        pred_df.drop(columns=[col for col in ['id', 'quality'] if col in pred_df.columns], inplace=True)

        # Validate input features
        missing = set(feature_names) - set(pred_df.columns)
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
        else:
            predictions = model.predict(pred_df)
            pred_df["predicted_quality"] = predictions

            st.write(pred_df.head())

            csv = pred_df.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv, "predicted_wine_quality.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Failed to process file: {e}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn")

import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load Trained Model
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model.joblib")

model = load_model()

# -----------------------
# Page Setup
# -----------------------
st.set_page_config(page_title="🌾 AI Crop Yield Predictor", page_icon="🌱", layout="centered")

# Custom Styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f1f8e9, #ffffff);
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2e7d32;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #388e3c;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<div class='title'>🌾 AI-based Crop Yield Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter soil, weather, and fertilizer details to get instant yield prediction (kg/ha)</div>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Dynamic Input Collection
# -----------------------
categorical_cols = model.named_steps["preprocessor"].transformers_[0][2]
numeric_cols = model.named_steps["preprocessor"].transformers_[1][2]

input_data = {}

with st.form("prediction_form"):
    st.markdown("### 📝 Input Details")

    # Render input widgets dynamically
    for col in categorical_cols:
        unique_values = []
        encoder = model.named_steps["preprocessor"].transformers_[0][1]
        try:
            idx = categorical_cols.index(col)
            unique_values = encoder.categories_[idx].tolist()
        except Exception:
            pass

        if unique_values:
            input_data[col] = st.selectbox(f"{col}", unique_values)
        else:
            input_data[col] = st.text_input(f"{col}")

    for col in numeric_cols:
        default_val = 0.0
        if "ph" in col.lower():
            default_val = 6.5
        elif "rain" in col.lower():
            default_val = 850.0
        elif "temp" in col.lower():
            default_val = 27.0

        input_data[col] = st.number_input(f"{col}", value=float(default_val))

    submitted = st.form_submit_button("🔮 Predict Crop Yield")

# -----------------------
# Prediction
# -----------------------
if submitted:
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]
    st.success("✅ Prediction Complete!")
    st.metric("🌾 Predicted Crop Yield (kg/ha)", f"{prediction:,.1f}")
    st.balloons()

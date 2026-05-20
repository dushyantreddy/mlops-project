import streamlit as st
import joblib
import pandas as pd
import boto3
import os

# ── Download Model from S3 if not present locally ──────────
def download_model_from_s3():
    bucket_name = "mlops-titanic-dushyant"   # ← your S3 bucket name
    s3_key = "model.joblib"                  # path inside the bucket
    local_path = "models/model.joblib"

    if not os.path.exists(local_path):
        st.info("Downloading model from S3...")
        os.makedirs("models", exist_ok=True)

        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-1"
        )
        s3.download_file(bucket_name, s3_key, local_path)
        st.success("Model downloaded successfully!")

download_model_from_s3()

# ── Load Model ─────────────────────────────────────────────
model = joblib.load("models/model.joblib")

# ── Page Config ────────────────────────────────────────────
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.write("Fill in the passenger details below to predict survival.")

# ── Input Form ─────────────────────────────────────────────
st.subheader("Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3],
    help="1 = First Class, 2 = Second Class, 3 = Third Class"
)

sex = st.radio("Sex", options=["Female", "Male"])
sex_encoded = 1 if sex == "Male" else 0

age = st.slider("Age", min_value=1, max_value=80, value=28)

sibsp = st.number_input(
    "Number of Siblings / Spouses aboard",
    min_value=0, max_value=8, value=0
)

parch = st.number_input(
    "Number of Parents / Children aboard",
    min_value=0, max_value=6, value=0
)

fare = st.number_input(
    "Ticket Fare (£)",
    min_value=0.0, max_value=520.0, value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    options=["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"]
)
embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
embarked_encoded = embarked_map[embarked]

# ── Predict ────────────────────────────────────────────────
if st.button("🔮 Predict Survival"):
    input_data = pd.DataFrame([[
        pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded
    ]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success("✅ This passenger would have **SURVIVED**")
    else:
        st.error("❌ This passenger would **NOT** have survived")

    st.write(f"**Survival Probability:** {probability[1]:.1%}")
    st.write(f"**Non-Survival Probability:** {probability[0]:.1%}")
    st.progress(float(probability[1]))
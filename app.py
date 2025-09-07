import streamlit as st
import numpy as np
import joblib

# load your trained model (pipeline or SVC + scaler)
model = joblib.load("svc_model.joblib")


FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

st.title("Breast Cancer Detection (SVC Model)")

inputs = []
cols = st.columns(3)
for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        val = st.number_input(f, value=0.0, format="%.4f")
        inputs.append(val)

if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    pred = model.predict(X)[0]
    st.write(f"### Prediction: {'Malignant' if pred == 1 else 'Benign'}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        st.progress(float(proba[1]))
        st.write(f"Probability → Benign: {proba[0]:.3f}, Malignant: {proba[1]:.3f}")

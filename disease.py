# Multi Disease Prediction – With Default Values, Dataset Preview & UI Fixes

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# --------------------------
# Model & Dataset Paths (EDIT IF NEEDED)
# --------------------------
MODEL_PARKINSON = r"C:\Users\katlee\nithi\parkinsons_model.joblib"
MODEL_LIVER = r"C:\Users\katlee\nithi\liver_model.joblib"
MODEL_KIDNEY = r"C:\Users\katlee\nithi\kidney_model.joblib"
SCALER_KIDNEY = r"C:\Users\katlee\nithi\minmax_scaler_kidney.joblib"

DATA_KIDNEY = r"C:\Users\katlee\nithi\kidney_disease_clean.csv"
DATA_LIVER = r"C:\Users\katlee\nithi\liver_disease_clean.csv"


# --------------------------
# Safe Loaders
# --------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except:
        return None

def safe_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None


parkinson_model = safe_load(MODEL_PARKINSON)
liver_model = safe_load(MODEL_LIVER)
kidney_model = safe_load(MODEL_KIDNEY)
kidney_scaler = safe_load(SCALER_KIDNEY)

df_kidney = safe_csv(DATA_KIDNEY)
df_liver = safe_csv(DATA_LIVER)

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="Multi Disease Prediction", layout="wide", page_icon="🩺")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/abstract-blur-hospital-clinic-medical-interior-background_293060-7298.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Multi Disease Prediction System")
st.write("Predict **Parkinson**, **Liver**, and **Kidney** diseases using machine learning.")


# --------------------------
# Sidebar Menu
# --------------------------
with st.sidebar:
    selected = option_menu(
        "Disease Prediction",
        ["🏥 Parkinson's", "💉 Liver", "🫘 Kidney"],
        icons=["activity", "droplet", "capsule"],
        default_index=0
    )


# --------------------------
# Parkinson's Disease Page
# --------------------------
if selected == "🏥 Parkinson's":

    st.header("🏥 Parkinson's Disease Prediction")
    st.write("Provide the voice measurement values.")

    cols = st.columns(5)

    defaults = {
        "fo": "119.992",
        "fhi": "157.302",
        "flo": "74.997",
        "Jitter_percent": "0.00784",
        "Jitter_Abs": "0.00007",
        "RAP": "0.00370",
        "PPQ": "0.00337",
        "DDP": "0.01110",
        "Shimmer": "0.04374",
        "Shimmer_dB": "0.426",
        "APQ3": "0.02182",
        "APQ5": "0.03130",
        "APQ": "0.02971",
        "DDA": "0.06545",
        "NHR": "0.02211",
        "HNR": "21.033",
        "RPDE": "0.414783",
        "DFA": "0.815285",
        "spread1": "-7.964984",
        "spread2": "0.154",
        "D2": "2.301442",
        "PPE": "0.284654"
    }

    parkinson_inputs = []
    for i, (key, val) in enumerate(defaults.items()):
        parkinson_inputs.append(cols[i % 5].text_input(key, value=val))

    if st.button("Predict Parkinson"):
        try:
            X = np.array([float(x) for x in parkinson_inputs]).reshape(1, -1)
            pred = parkinson_model.predict(X)
            if pred[0] == 1:
                st.error("🛑 Parkinson Disease Detected")
            else:
                st.success("✅ No Parkinson Disease")
        except:
            st.error("Please enter valid numeric values.")


# --------------------------
# Liver Disease Page
# --------------------------
elif selected == "💉 Liver":

    st.header("💉 Liver Disease Prediction 🧪")
    st.write("Enter the medical values for liver disease prediction.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.text_input("Age", value="45")
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    with c3:
        total_bilirubin = st.text_input("Total Bilirubin", value="1.0")
    with c4:
        direct_bilirubin = st.text_input("Direct Bilirubin", value="0.5")

    with c1:
        alkaline_phosphatase = st.text_input("Alkaline Phosphatase", value="200")
    with c2:
        alt = st.text_input("ALT", value="35")
    with c3:
        ast = st.text_input("AST", value="40")
    with c4:
        total_proteins = st.text_input("Total Proteins", value="6.8")

    with c1:
        albumin = st.text_input("Albumin", value="3.5")
    with c2:
        agr = st.text_input("A/G Ratio", value="1.1")

    if st.button("Predict Liver Disease"):
        try:
            gender_num = 1 if gender == "Male" else 0

            vals = [
                gender_num, age, total_bilirubin, direct_bilirubin,
                alkaline_phosphatase, alt, ast, total_proteins,
                albumin, agr
            ]

            X = np.array([float(v) for v in vals]).reshape(1, -1)
            pred = liver_model.predict(X)

            if pred[0] == 1:
                st.error("🛑 Liver Disease Detected")
            else:
                st.success("✅ No Liver Disease")
        except:
            st.error("Enter valid numeric values.")

    st.subheader("📊 Liver Dataset")
    if df_liver is not None:
        if st.button("Show Liver Dataset"):
            st.dataframe(df_liver)


# --------------------------
# Kidney Disease Page
# --------------------------
elif selected == "🫘 Kidney":

    st.header("🫘 Kidney Disease Prediction")
    st.write("Enter the medical values for kidney disease detection.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.text_input("Age", value="45")
    with c2:
        bp = st.text_input("Blood Pressure", value="80")
    with c3:
        sg = st.text_input("Specific Gravity", value="1.02")
    with c4:
        al = st.text_input("Albumin", value="1")

    with c1:
        su = st.text_input("Sugar", value="0")
    with c2:
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], index=0)
    with c3:
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"], index=0)
    with c4:
        pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], index=1)

    with c1:
        ba = st.selectbox("Bacteria", ["present", "notpresent"], index=1)
    with c2:
        bgr = st.text_input("Blood Glucose Random", value="120")
    with c3:
        bu = st.text_input("Blood Urea", value="40")
    with c4:
        sc = st.text_input("Serum Creatinine", value="1.2")

    with c1:
        sod = st.text_input("Sodium", value="135")
    with c2:
        pot = st.text_input("Potassium", value="4.5")
    with c3:
        hemo = st.text_input("Hemoglobin", value="15")
    with c4:
        pcv = st.text_input("Packed Cell Volume", value="44")

    with c1:
        wbc = st.text_input("WBC Count", value="8000")
    with c2:
        rbc_count = st.text_input("RBC Count", value="5")
    with c3:
        htn = st.selectbox("Hypertension", ["yes", "no"], index=1)
    with c4:
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=1)

    with c1:
        cad = st.selectbox("Coronary Artery Disease", ["yes", "no"], index=1)
    with c2:
        appet = st.selectbox("Appetite", ["good", "poor"], index=0)
    with c3:
        pe = st.selectbox("Pedal Edema", ["yes", "no"], index=1)
    with c4:
        ane = st.selectbox("Anemia", ["yes", "no"], index=1)

    if st.button("Predict Kidney Disease"):
        try:

            rbc_num = 0 if rbc == "normal" else 1
            pc_num = 0 if pc == "normal" else 1
            pcc_num = 1 if pcc == "present" else 0
            ba_num = 1 if ba == "present" else 0

            yesno = lambda x: 1 if x == "yes" else 0

            vals = [
                age, bp, sg, al, su, rbc_num, pc_num, pcc_num, ba_num,
                bgr, bu, sc, sod, pot, hemo, pcv, wbc, rbc_count,
                yesno(htn), yesno(dm), yesno(cad), 1 if appet=="good" else 0,
                yesno(pe), yesno(ane)
            ]

            X = np.array([float(v) for v in vals]).reshape(1, -1)
            X_scaled = kidney_scaler.transform(X)

            pred = kidney_model.predict(X_scaled)

            if pred[0] == 1:
                st.error("🛑 Kidney Disease Detected")
            else:
                st.success("✅ No Kidney Disease")
        except:
            st.error("Enter valid numeric values.")

    st.subheader("📊 Kidney Dataset")
    if df_kidney is not None:
        if st.button("Show Kidney Dataset"):
            st.dataframe(df_kidney)

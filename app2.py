import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ==============================
# 1️⃣ APP TITLE & DESCRIPTION
# ==============================
st.set_page_config(page_title="Smart Loan Approval System")

st.title("🏦 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict "
    "whether a loan should be **Approved or Rejected**."
)

# ==============================
# LOAD & PREPARE DATA
# ==============================
df = pd.read_csv("train_loan.csv")

# Drop Loan_ID
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
for col in df.select_dtypes(include=["int64", "float64"]):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=["object"]):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Feature engineering
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]):
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

feature_order = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================
# TRAIN SVM MODELS
# ==============================
svm_models = {
    "Linear SVM": SVC(kernel="linear", probability=True),
    "Polynomial SVM": SVC(kernel="poly", degree=3, probability=True),
    "RBF SVM": SVC(kernel="rbf", probability=True),
}

for model in svm_models.values():
    model.fit(X_train_scaled, y_train)

# ==============================
# 2️⃣ INPUT SECTION (SIDEBAR)
# ==============================
st.sidebar.header("📋 Applicant Details")

applicant_income = st.sidebar.number_input(
    "Applicant Income", min_value=0, value=5000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=0, value=150
)

credit_history = st.sidebar.selectbox(
    "Credit History", ["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status", ["Salaried", "Self Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# ==============================
# 3️⃣ MODEL SELECTION
# ==============================
kernel_choice = st.sidebar.radio(
    "Select SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# Encode inputs
credit_val = 1 if credit_history == "Yes" else 0
employment_val = 1 if employment_status == "Self Employed" else 0
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

# ==============================
# CREATE INPUT DATA
# ==============================
input_data = pd.DataFrame([{
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": 0,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": 360,
    "Credit_History": credit_val,
    "Gender": 1,
    "Married": 1,
    "Dependents": 0,
    "Education": 1,
    "Self_Employed": employment_val,
    "Property_Area": property_map[property_area],
    "TotalIncome": applicant_income
}])

input_data = input_data[feature_order]
input_scaled = scaler.transform(input_data)

# ==============================
# 4️⃣ PREDICTION BUTTON
# ==============================
if st.button("🔍 Check Loan Eligibility"):
    model = svm_models[kernel_choice]

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max() * 100

    # ==============================
    # 5️⃣ OUTPUT SECTION
    # ==============================
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    # ==============================
    # 6️⃣ BUSINESS EXPLANATION
    # ==============================
    st.subheader("📌 Business Explanation")

    if prediction == 1:
        st.write(
            "Based on the applicant’s **credit history and income pattern**, "
            "the applicant is **likely to repay the loan on time**."
        )
    else:
        st.write(
            "Based on the applicant’s **credit history and income pattern**, "
            "the applicant is **unlikely to repay the loan**, making it risky."
        )

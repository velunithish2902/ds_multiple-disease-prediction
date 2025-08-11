import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# --------------------------
# Load datasets
# --------------------------
liver_df = pd.read_csv(r'C:\python\new\venv\venv\indian_liver_patient - indian_liver_patient (1).csv')
kidney_df = pd.read_csv(r'C:\python\new\venv\venv\kidney_disease - kidney_disease (1).csv')
parkinsons_df = pd.read_csv(r'C:\python\new\venv\venv\parkinsons - parkinsons (1).csv')

st.title("Disease Prediction Dashboard")

# --------------------------
# Sidebar selection
# --------------------------
st.sidebar.title("Prediction Section")
dataset_choice = st.sidebar.selectbox(
    "Select Disease for Prediction",
    ("Liver Disease", "Kidney Disease", "Parkinson's Disease")
)

# --------------------------
# LIVER DISEASE
# --------------------------
if dataset_choice == "Liver Disease":
    st.subheader("Liver Disease Dataset")
    st.dataframe(liver_df.head())

    liver_df.columns = liver_df.columns.str.strip()
    liver_df.fillna(liver_df.median(numeric_only=True), inplace=True)
    for col in liver_df.select_dtypes(include='object'):
        liver_df[col] = liver_df[col].fillna(liver_df[col].mode()[0])
        liver_df[col] = LabelEncoder().fit_transform(liver_df[col])

    X = liver_df.drop('Dataset', axis=1)
    y = liver_df['Dataset']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=4, min_samples_leaf=2,
        class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    st.subheader(f"Liver Disease Accuracy: {accuracy:.2f}%")

    st.subheader("Make a Prediction (Liver Disease)")
    input_data = []
    liver_features = list(liver_df.drop('Dataset', axis=1).columns)

    for feature in liver_features:
        if feature.lower() == 'gender':
            gender_value = st.selectbox("Gender", options=['Male', 'Female'])
            gender_value = 1 if gender_value == 'Male' else 0
            input_data.append(gender_value)
        else:
            value = st.number_input(f"Enter {feature}", value=0.0)
            input_data.append(value)

    if st.button("Predict (Liver)"):
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        result = "Disease Detected" if prediction == 1 else "No Disease"
        st.success(f"Prediction: {result}")

# --------------------------
# KIDNEY DISEASE
# --------------------------
elif dataset_choice == "Kidney Disease":
    st.subheader("Kidney Disease Dataset")
    st.dataframe(kidney_df.head())

    # Fill bgr with median
    median_bgr = kidney_df['bgr'].median()
    kidney_df['bgr'] = kidney_df['bgr'].fillna(median_bgr)

    # Drop null rows
    kidney_df.dropna(inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = list(kidney_df.select_dtypes(include='object').columns)
    for col in categorical_cols:
        kidney_df[col] = kidney_df[col].fillna(kidney_df[col].mode()[0])
        le = LabelEncoder()
        kidney_df[col] = le.fit_transform(kidney_df[col])
        label_encoders[col] = le  # store encoder for later use

    X = kidney_df.drop('classification', axis=1)
    y = LabelEncoder().fit_transform(kidney_df['classification'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    st.subheader(f"Kidney Disease Accuracy: {accuracy:.2f}%")

    st.subheader("Make a Prediction (Kidney Disease)")

    # Build input dynamically
    input_data = []
    for feature in X.columns:
        if feature in label_encoders:  # categorical
            selected_val = st.selectbox(f"{feature}", label_encoders[feature].classes_)
            encoded_val = label_encoders[feature].transform([selected_val])[0]
            input_data.append(encoded_val)
        else:  # numeric
            val = st.number_input(f"Enter {feature}", value=0.0)
            input_data.append(val)

    if st.button("Predict (Kidney)"):
        prediction = model.predict([input_data])[0]
        result = "Disease Detected" if prediction == 1 else "No Disease"
        st.success(f"Prediction: {result}")

# --------------------------
# PARKINSON'S DISEASE
# --------------------------
else:
    st.subheader("Parkinson's Disease Dataset")
    st.dataframe(parkinsons_df.head())

    if 'name' in parkinsons_df.columns:
        parkinsons_df.drop('name', axis=1, inplace=True)

    parkinsons_df.fillna(parkinsons_df.median(numeric_only=True), inplace=True)
    for col in parkinsons_df.select_dtypes(include='object'):
        parkinsons_df[col] = parkinsons_df[col].fillna(parkinsons_df[col].mode()[0])
        parkinsons_df[col] = LabelEncoder().fit_transform(parkinsons_df[col])

    X = parkinsons_df.drop('status', axis=1)
    y = parkinsons_df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    st.subheader(f"Parkinson's Disease Accuracy: {accuracy:.2f}%")

    st.subheader("Make a Prediction (Parkinson's Disease)")
    input_data = []
    parkinsons_features = list(X.columns)

    for feature in parkinsons_features:
        value = st.number_input(f"Enter {feature}", value=0.0)
        input_data.append(value)

    if st.button("Predict (Parkinson's)"):
        prediction = model.predict([input_data])[0]
        result = "Disease Detected" if prediction == 1 else "No Disease"
        st.success(f"Prediction: {result}")

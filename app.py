import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓", layout="wide")

DATA_FILE = "student_placement_prediction_dataset_2026 (2).csv"

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    # Same cleaning approach used in the project notebook
    df = df.drop(columns=["student_id", "salary_package_lpa"], errors="ignore")
    df = df.dropna(subset=["placement_status"])
    df = df.drop_duplicates()

    X = df.drop(columns=["placement_status"])
    y = df["placement_status"].map({"Not Placed": 0, "Placed": 1})

    categorical_features = [
        "gender", "branch", "college_tier", "volunteer_experience"
    ]

    X = pd.get_dummies(X, columns=categorical_features)
    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest configuration used in the project improvement stage
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler, feature_columns, df

try:
    model, scaler, feature_columns, df = train_model()
except Exception as e:
    st.error("The application could not load the dataset.")
    st.code(str(e))
    st.stop()

st.title("🎓 Student Placement Prediction System")
st.write(
    "Enter student details to get a Machine Learning based placement prediction."
)

st.info("This prediction is based on patterns learned from the dataset and is not a guarantee of actual placement.")

with st.form("placement_form"):
    st.subheader("Student Details")
    values = {}

    numeric_features = [
        "age", "cgpa", "internships_count", "projects_count",
        "certifications_count", "coding_skill_score", "aptitude_score",
        "communication_skill_score", "logical_reasoning_score",
        "hackathons_participated", "github_repos", "linkedin_connections",
        "mock_interview_score", "attendance_percentage", "backlogs",
        "extracurricular_score", "leadership_score", "sleep_hours",
        "study_hours_per_day"
    ]

    categorical_features = [
        "gender", "branch", "college_tier", "volunteer_experience"
    ]

    c1, c2 = st.columns(2)

    with c1:
        for feature in numeric_features[:10]:
            if feature in df.columns:
                col = df[feature]
                default = float(col.median())
                min_val = float(col.min())
                max_val = float(col.max())
                step = 0.01 if feature in ["cgpa", "sleep_hours", "study_hours_per_day"] else 1.0
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    step=step
                )

    with c2:
        for feature in numeric_features[10:]:
            if feature in df.columns:
                col = df[feature]
                default = float(col.median())
                min_val = float(col.min())
                max_val = float(col.max())
                step = 0.01 if feature in ["sleep_hours", "study_hours_per_day"] else 1.0
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    step=step
                )

    st.subheader("Categorical Details")
    c1, c2 = st.columns(2)
    for idx, feature in enumerate(categorical_features):
        if feature in df.columns:
            options = sorted(df[feature].dropna().astype(str).unique().tolist())
            if not options:
                continue
            with (c1 if idx % 2 == 0 else c2):
                values[feature] = st.selectbox(
                    feature.replace("_", " ").title(), options
                )

    submitted = st.form_submit_button("Predict Placement")

if submitted:
    input_df = pd.DataFrame([values])

    # Apply the same one-hot encoding used during training
    input_df = pd.get_dummies(input_df, columns=categorical_features)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.divider()
    if int(prediction) == 1:
        st.success("Prediction: Likely to be Placed")
    else:
        st.warning("Prediction: Not Likely to be Placed")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1] * 100
        st.metric("Estimated Placement Probability", f"{probability:.2f}%")

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance & Academic Risk", page_icon="🎓", layout="centered")

# Load trained models
marks_model = joblib.load("final_marks_model.pkl")
risk_model = joblib.load("academic_risk_model.pkl")

st.title("🎓 Student Performance Prediction & Academic Risk")
st.write("Enter student academic and behavioral metrics below to predict Final Marks and Academic Risk level.")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance Percentage (%)", 0.0, 100.0, 78.0)
    study_hrs = st.slider("Study Hours Per Day", 0.0, 15.0, 4.0, step=0.1)
    assignment = st.slider("Assignment Score", 0.0, 100.0, 72.0)
    internal = st.slider("Internal Marks", 0.0, 100.0, 68.0)
    prev_sem = st.slider("Previous Semester Marks", 0.0, 100.0, 66.0)

with col2:
    participation = st.slider("Class Participation (1 - 5)", 1, 5, 3)
    sleep = st.slider("Average Sleep Hours", 3.0, 12.0, 7.0, step=0.1)
    internet = st.selectbox("Internet Access", ["Limited", "Average", "Good"], index=2)
    extra = st.radio("Extracurricular Activity", ["Yes", "No"], index=1)

if st.button("Predict Performance & Risk", type="primary", use_container_width=True):
    internet_map = {"Limited": 0, "Average": 1, "Good": 2}
    extra_map = {"No": 0, "Yes": 1}

    input_df = pd.DataFrame([{
        "Attendance_Percentage": attendance,
        "Study_Hours_Per_Day": study_hrs,
        "Assignment_Score": assignment,
        "Internal_Marks": internal,
        "Previous_Semester_Marks": prev_sem,
        "Class_Participation": participation,
        "Average_Sleep_Hours": sleep,
        "Internet_Access": internet_map[internet],
        "Extracurricular_Activity": extra_map[extra]
    }])

    pred_marks = marks_model.predict(input_df)[0]
    pred_risk = risk_model.predict(input_df)[0]

    st.divider()
    res1, res2 = st.columns(2)
    res1.metric("Predicted Final Marks", f"{pred_marks:.2f} / 100")
    res2.metric("Academic Risk Level", f"{pred_risk} Risk")

    if pred_risk == "High":
        st.error("⚠️ High Academic Risk: Immediate academic mentoring and attendance improvement recommended.")
    elif pred_risk == "Medium":
        st.warning("⚡ Medium Academic Risk: Consistent study hours and assignment tracking advised.")
    else:
        st.success("✅ Low Academic Risk: Student is performing well.")

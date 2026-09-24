import streamlit as st
import pandas as pd
import joblib

# Load trained model

model = joblib.load("models/employee_attrition_model.pkl")


# Page configuration

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Attrition Prediction")
st.write(
    "Enter employee information below to predict whether "
    "the employee is likely to leave the organization."
)

st.divider()


# Employee Information

st.header("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    BusinessTravel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )

    DailyRate = st.number_input(
        "Daily Rate",
        min_value=100,
        max_value=1500,
        value=800
    )

    Department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    DistanceFromHome = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    Education = st.selectbox(
        "Education Level",
        [1, 2, 3, 4, 5]
    )

    EducationField = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    EnvironmentSatisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    HourlyRate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=65
    )


with col2:

    JobInvolvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4]
    )

    JobLevel = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    JobRole = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    JobSatisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=100,
        max_value=20000,
        value=5000
    )

    MonthlyRate = st.number_input(
        "Monthly Rate",
        min_value=2000,
        max_value=27000,
        value=14000
    )

    NumCompaniesWorked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    OverTime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    PercentSalaryHike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        max_value=30,
        value=15
    )


with col3:

    PerformanceRating = st.selectbox(
        "Performance Rating",
        [1, 2, 3, 4]
    )

    RelationshipSatisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1, 2, 3, 4]
    )

    StockOptionLevel = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3]
    )

    TotalWorkingYears = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=50,
        value=8
    )

    TrainingTimesLastYear = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=10,
        value=3
    )

    WorkLifeBalance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )

    YearsAtCompany = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=50,
        value=5
    )

    YearsInCurrentRole = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=3
    )

    YearsSinceLastPromotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=20,
        value=1
    )

    YearsWithCurrManager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=3
    )


# Prediction

st.divider()

if st.button("🔮 Predict Attrition", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [Age],
        "BusinessTravel": [BusinessTravel],
        "DailyRate": [DailyRate],
        "Department": [Department],
        "DistanceFromHome": [DistanceFromHome],
        "Education": [Education],
        "EducationField": [EducationField],
        "EnvironmentSatisfaction": [EnvironmentSatisfaction],
        "Gender": [Gender],
        "HourlyRate": [HourlyRate],
        "JobInvolvement": [JobInvolvement],
        "JobLevel": [JobLevel],
        "JobRole": [JobRole],
        "JobSatisfaction": [JobSatisfaction],
        "MaritalStatus": [MaritalStatus],
        "MonthlyIncome": [MonthlyIncome],
        "MonthlyRate": [MonthlyRate],
        "NumCompaniesWorked": [NumCompaniesWorked],
        "OverTime": [OverTime],
        "PercentSalaryHike": [PercentSalaryHike],
        "PerformanceRating": [PerformanceRating],
        "RelationshipSatisfaction": [RelationshipSatisfaction],
        "StockOptionLevel": [StockOptionLevel],
        "TotalWorkingYears": [TotalWorkingYears],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
        "WorkLifeBalance": [WorkLifeBalance],
        "YearsAtCompany": [YearsAtCompany],
        "YearsInCurrentRole": [YearsInCurrentRole],
        "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Employee is predicted to leave.")
    else:
        st.success("✅ Employee is predicted to stay.")

    st.metric(
        "Predicted Attrition Probability",
        f"{probability * 100:.2f}%"
    )
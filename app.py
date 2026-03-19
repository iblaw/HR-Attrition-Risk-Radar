import streamlit as st
import pandas as pd
import joblib

# 1. APP SETUP & STYLING
st.set_page_config(page_title="HR Flight Risk Radar", page_icon="🚨", layout="centered")
st.title("HR Attrition Risk Radar 🚨")
st.write("Enter an employee's details below to predict their likelihood of leaving the company.")


# 2. LOAD THE MODEL (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    model = joblib.load('hr_flight_risk_model_v1.pkl')
    scaler = joblib.load('hr_scaler.pkl')
    # This will now load as a simple dictionary
    baseline = joblib.load('hr_baseline_employee.pkl')
    cols = joblib.load('hr_model_columns.pkl')
    return model, scaler, baseline, cols

loaded_model, scaler, baseline_employee, expected_columns = load_model()

# 3. THE UI FORM (Creating input columns for a clean look)
st.subheader("Employee Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
    distance = st.number_input("Distance From Home (miles)", min_value=1, max_value=100, value=10)

with col2:
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
    env_sat = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
    new_manager = st.number_input("Years With Current Manager", min_value=0, max_value=40, value=2)

# 4. THE PREDICTION ENGINE
st.divider()
if st.button("Predict Flight Risk", type="primary", use_container_width=True):

    raw_data = {
        'Age': age,
        'MonthlyIncome': income,
        'DistanceFromHome': distance,
        'EnvironmentSatisfaction': env_sat,
        'YearsWithCurrManager': new_manager,
        'OverTime_Yes': 1 if overtime == "Yes" else 0
    }

    # THE NEW ALIGNMENT TRICK: Start with a perfectly average employee!
    df = pd.DataFrame([baseline_employee], columns=expected_columns)

    # Overwrite the average stats with the specific 6 stats HR typed in
    for key, value in raw_data.items():
        if key in df.columns:
            df.at[0, key] = value

    # Scale and Predict
    df_scaled = scaler.transform(df)
    risk_prob = loaded_model.predict_proba(df_scaled)[0][1]

    # 5. DISPLAY RESULTS
    st.subheader(f"Calculated Flight Risk: {risk_prob * 100:.1f}%")

    # Visual Progress Bar
    st.progress(float(risk_prob))

    # Business Logic / Alerts
    if risk_prob > 0.30:
        st.error("🚨 HIGH FLIGHT RISK DETECTED")
        st.write(
            "**Action Required:** Schedule a stay interview immediately. Prioritize discussions around workload (overtime) and compensation parity.")
    else:
        st.success("✅ Employee Status: Safe/Stable")
        st.write("**Action Required:** None. Continue standard management check-ins.")
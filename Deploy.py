import joblib
import pandas as pd

# 1. UN-PICKLE THE MODEL AND THE COLUMNS
loaded_model = joblib.load('hr_flight_risk_model_v1.pkl')
expected_columns = joblib.load('hr_model_columns.pkl') # <-- The Fix!

# 2. THE NEW EMPLOYEE (Raw input)
raw_employee_data = {
    'Age': 22,
    'DailyRate': 0,
    'DistanceFromHome': 2000,
    'EnvironmentSatisfaction': 1,
    'MonthlyIncome': 1500,
    'OverTime_Yes': 0,
    'YearsWithCurrManager': 0,
    'JobRole_Sales Executive': 0,
    'MaritalStatus_Single': 0
}

# 3. THE ALIGNMENT TRICK
# Create the empty dataframe using our safely loaded column names
new_employee_df = pd.DataFrame(0, index=[0], columns=expected_columns)

# Loop through and fill in the data
for key, value in raw_employee_data.items():
    if key in new_employee_df.columns:
        new_employee_df.at[0, key] = value

# 4. RUN THE RADAR
probabilities = loaded_model.predict_proba(new_employee_df)
risk_of_leaving = probabilities[0][1]

print(f"Calculated Flight Risk: {risk_of_leaving * 100:.1f}%\n")

if risk_of_leaving > 0.30:
    print("🚨 SYSTEM ALERT: High Flight Risk Detected!")
else:
    print("✅ Employee Status: Safe/Stable.")
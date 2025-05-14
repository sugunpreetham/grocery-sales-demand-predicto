import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the trained model
model = joblib.load('C:/Users/sugun/Desktop/sales_demand_model.pkl')


# App title
st.title("🛒 Grocery Sales Demand Predictor")

# Input form
st.header("📥 Enter Input Features")
store_id = st.number_input("Store ID", min_value=1)
product_id = st.number_input("Product ID", min_value=1)
week = st.number_input("Week Number", min_value=1, max_value=52)

# Date picker instead of separate day/month/year inputs
selected_date = st.date_input("Select Date", datetime.date.today())
day = selected_date.day
month = selected_date.month
year = selected_date.year

# Predict button
if st.button("🔍 Predict Demand"):
    input_data = pd.DataFrame([{
        'Store_ID': store_id,
        'Product_ID': product_id,
        'Week': week,
        'Day': day,
        'Month': month,
        'Year': year
    }])
    
    prediction = model.predict(input_data)
    predicted_value = prediction[0]

    st.success(f"📦 Predicted Sales: {predicted_value:.2f} units")

    # Optional download section
    result_df = pd.DataFrame([{
        'Store_ID': store_id,
        'Product_ID': product_id,
        'Week': week,
        'Day': day,
        'Month': month,
        'Year': year,
        'Predicted_Sales': predicted_value
    }])

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Result as CSV",
        data=csv,
        file_name='sales_prediction_result.csv',
        mime='text/csv',
    )

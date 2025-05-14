# Save this as train_model.py and run it oncepip

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Sample data (you can replace with your CSV file)
data = {
    'Store_ID': [1, 1, 2, 2],
    'Product_ID': [101, 102, 101, 102],
    'Week': [1, 2, 1, 2],
    'Date': ['2023-01-01', '2023-01-08', '2023-01-01', '2023-01-08'],
    'Sales': [200, 210, 180, 190]
}
df = pd.DataFrame(data)

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

X = df[['Store_ID', 'Product_ID', 'Week', 'Day', 'Month', 'Year']]
y = df['Sales']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
import joblib
import os

# Save model
model_path = os.path.abspath('sales_demand_model.pkl')
joblib.dump(model, model_path)
print(f"✅ Model saved at: {model_path}")


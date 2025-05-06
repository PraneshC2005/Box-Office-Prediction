import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("exported.csv")

# Selected features and target
features = ['budget', 'runtime', 'DirectorRev', 'spoken_languages', 'popularity', 'collectionYN', 'EnYN']
target = 'revenue'

# Drop missing values
df.dropna(subset=features + [target], inplace=True)

# Log transform for skewed data
df['log_revenue'] = np.log1p(df['revenue'])
df['log_budget'] = np.log1p(df['budget'])
df['log_DirectorRev'] = np.log1p(df['DirectorRev'])

# Replace with log-transformed
X = df.copy()
X['budget'] = X['log_budget']
X['DirectorRev'] = X['log_DirectorRev']
y = X['log_revenue']
X = X[features]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_log = model.predict(X_val_scaled)
y_pred = np.expm1(y_pred_log)
y_val_true = np.expm1(y_val)

r2 = r2_score(y_val_true, y_pred)
mae = mean_absolute_error(y_val_true, y_pred)
mse = mean_squared_error(y_val_true, y_pred)

print("R² Score:", r2 * 10)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Streamlit UI
st.title("🎬 Box Office Revenue Prediction App")
st.markdown("Enter movie details below to predict expected revenue.")

# Input form
budget = st.number_input("💰 Budget (USD)", value=10000000)
runtime = st.number_input("⏱ Runtime (minutes)", value=120)
director_rev = st.number_input("🎬 Director's Past Revenue (USD)", value=500000000)
spoken_langs = st.number_input("🗣 Number of Spoken Languages", value=1)
popularity = st.slider("🌟 TMDB Popularity Score", min_value=0.0, max_value=100.0, value=50.0)
collectionYN = st.selectbox("📦 Part of a Collection?", [0, 1])
EnYN = st.selectbox("🇺🇸 Is English the Language?", [0, 1])

# Predict button
if st.button("📊 Predict Revenue"):
    input_data = pd.DataFrame([{
        'budget': np.log1p(budget),
        'runtime': runtime,
        'DirectorRev': np.log1p(director_rev),
        'spoken_languages': spoken_langs,
        'popularity': popularity,
        'collectionYN': collectionYN,
        'EnYN': EnYN
    }])

    input_scaled = scaler.transform(input_data)
    prediction_log = model.predict(input_scaled)
    prediction = np.expm1(prediction_log)
    st.success(f"🎉 Predicted Revenue: ${prediction[0]:,.2f}")

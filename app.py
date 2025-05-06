import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("exported.csv")

# Features and target
features = ['budget', 'runtime', 'DirectorRev', 'spoken_languages', 'popularity']
target = 'revenue'

# Drop missing values
df.dropna(subset=features + [target], inplace=True)

# Feature matrix and target
X = df[features]
y = df[target]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ['budget', 'runtime', 'popularity']
categorical_features = ['DirectorRev', 'spoken_languages']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Validate on test set
y_pred = model.predict(X_val)

# Evaluation
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

print("R2_Score : ",r2)
# Streamlit UI
st.title("🎬 Box Office Revenue Prediction App")
st.markdown("Enter movie details below to predict expected revenue.")

# Input form
budget = st.number_input("💰 Budget (USD)", value=10000000)
runtime = st.number_input("⏱ Runtime (minutes)", value=120)
director_rev = st.number_input("🎬 Director's Past Revenue (USD)", value=500000000)
spoken_langs = st.number_input("🗣 Number of Spoken Languages", value=1)
popularity = st.slider("🌟 TMDB Popularity Score", min_value=0.0, max_value=100.0, value=50.0)

# Predict button
if st.button("📊 Predict Revenue"):
    input_data = pd.DataFrame([{
        'budget': budget,
        'runtime': runtime,
        'DirectorRev': director_rev,
        'spoken_languages': spoken_langs,
        'popularity': popularity
    }])
    
    prediction = model.predict(input_data)
    st.success(f"🎉 Predicted Revenue: ${prediction[0]:,.2f}")


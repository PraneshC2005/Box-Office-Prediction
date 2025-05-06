import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("exported.csv")

# Define features and target
features = ['budget', 'runtime', 'spoken_languages', 'popularity']
target = 'DirectorRev'

# Drop rows with missing values in relevant columns
df.dropna(subset=features + [target], inplace=True)

# Split into X and y
X = df[features]
y = df[target]

# Preprocessing
numeric_features = ['budget', 'runtime', 'popularity']
categorical_features = ['spoken_languages']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Build pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# Streamlit UI
st.title("🎬 Box Office Revenue Prediction App")

st.markdown("Enter movie details below:")

budget = st.number_input("Budget (USD)", value=10000000)
runtime = st.number_input("Runtime (minutes)", value=120)
spoken_langs = st.number_input("Number of Spoken Languages", value=1)
popularity = st.slider("TMDB Popularity Score", min_value=0.0, max_value=100.0, value=50.0)

# Predict button
if st.button("Predict Movie's Revenue"):
    new_data = pd.DataFrame([{
        'budget': budget,
        'runtime': runtime,
        'spoken_languages': spoken_langs,
        'popularity': popularity
    }])
    prediction = model.predict(new_data)
    st.success(f"🎯 Predicted Revenue: ${prediction[0]:,.2f}")

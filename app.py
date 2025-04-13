import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt


model = load_model('wine_quality_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine's characteristics below to predict its quality (range: 3 to 8).")


feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
       'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
       'sulphates', 'alcohol']


user_input = []
st.subheader("Wine Characteristics")
for feature in feature_names:
    value = st.number_input(f"{feature.title()}", value=0.0, format="%.4f")
    user_input.append(value)


if st.button("🔍 Predict Quality"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    predicted_class = prediction.argmax(axis=1)[0]
    
    st.success(f"🎯 Predicted Wine Quality: **{predicted_class+3}**")
    print(prediction)
    print(prediction.argmax(axis=1))
    
    
    quality_labels = list(range(3, 9))
    probs_df = pd.DataFrame(prediction, columns=quality_labels).T
    probs_df.columns = ["Probability"]
    probs_df["Probability"] = probs_df["Probability"].round(2)
    st.subheader("📊 Prediction Probabilities")
    st.dataframe(probs_df)

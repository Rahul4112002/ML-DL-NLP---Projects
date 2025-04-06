# Import necessary libraries
import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app layout
st.title("Music Popularity Prediction")

# Input fields for user to enter parameters
energy = st.slider("Energy", 0.0, 1.0, 0.5)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
tempo = st.number_input("Tempo (BPM)", min_value=0.0, step=0.1)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)

# Button to make prediction
if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[energy, valence, danceability, loudness, acousticness, tempo, speechiness, liveness]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction
    st.write(f'Predicted Popularity: {prediction[0]:.2f}')

# Run the app
if __name__ == '__main__':
    st.write("Enter the details above and click 'Predict' to see the estimated popularity.")
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('Real_Estate.csv')
X = data[['House age', 'Distance to the nearest MRT station', 'Number of convenience stores']]
y = data['House price of unit area']

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit app layout
st.title("Real Estate Price Prediction")

# Input fields for user to enter parameters
house_age = st.number_input("House Age (in years)", min_value=0.0, step=0.1)
distance_mrt = st.number_input("Distance to MRT (in meters)", min_value=0.0, step=1.0)
num_stores = st.number_input("Number of Convenience Stores", min_value=0, step=1)

# Button to make prediction
if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[house_age, distance_mrt, num_stores]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f'Predicted House Price per Unit Area: ${prediction[0]:.2f}')

# Run the app
if __name__ == '__main__':
    st.write("Enter the details above and click 'Predict' to see the estimated price.")
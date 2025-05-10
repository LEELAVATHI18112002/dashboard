import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('random_forest_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('scaler_y.pkl', 'rb') as scaler_y_file:
    scaler_y = pickle.load(scaler_y_file)

# Example input for prediction
data = np.array([[2,22]])  # Replace with your actual input data

# Apply the same scaling as during training
scaled_data = scaler.transform(data)

# Make a prediction
prediction = classifier.predict(scaled_data)

# Inverse transform the prediction
prediction_original_scale = scaler_y.inverse_transform(prediction.reshape(1, -1)).flatten()

# Print the predicted value in original scale
print(f"Predicted value: {prediction_original_scale[0]}")

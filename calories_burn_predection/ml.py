import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load Dataset
data = pd.read_csv('calories.csv')

# Preprocess Data
selected_features = ["Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
X = data[selected_features]
y = data["Calories"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "calorie_model.pkl")

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model saved successfully! Mean Squared Error: {mse}")

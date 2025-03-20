from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("calorie_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    calories = None

    if request.method == "POST":
        try:
            # Retrieve form data
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            duration = float(request.form['duration'])
            heart_rate = float(request.form['heart_rate'])
            body_temp = float(request.form['body_temp'])

            # Prepare data for prediction
            input_data = np.array([[height, weight, duration, heart_rate, body_temp]])
            
            # Make prediction
            calories = model.predict(input_data)[0]

        except Exception as e:
            calories = f"Error: {str(e)}"

    return render_template("index.html", calories=calories)

if __name__ == "__main__":
    app.run(debug=True)

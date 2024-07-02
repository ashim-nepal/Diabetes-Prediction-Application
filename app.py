import numpy as np
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Loading the trained model
model_filename = 'gbc_diebetes_prediction_model.pkl'
model = joblib.load(model_filename)

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data and convert to DataFrame
    form_data = request.form.to_dict()

    # Columns in the same sequential order as the model
    columns = ['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    input_data = pd.DataFrame([form_data], columns=columns)

    # Convert numeric fields to float types
    input_data['age'] = input_data['age'].astype(int)
    input_data['smoking_history'] = input_data['smoking_history'].astype(int)
    input_data['bmi'] = input_data['bmi'].astype(float)
    input_data['HbA1c_level'] = input_data['HbA1c_level'].astype(float)
    input_data['blood_glucose_level'] = input_data['blood_glucose_level'].astype(int)

    # Convert DataFrame to numpy array

    # Prediction making
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        return str(e)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

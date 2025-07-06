from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)
model = pickle.load(open('model/heart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form['Age']),
            float(request.form['Sex']),
            float(request.form['ChestPainType']),
            float(request.form['RestingBP']),
            float(request.form['Cholesterol']),
            float(request.form['FastingBS']),
            float(request.form['RestingECG']),
            float(request.form['MaxHR']),
            float(request.form['ExerciseAngina']),
            float(request.form['Oldpeak']),
            float(request.form['ST_Slope'])
        ]
        prediction = model.predict([input_features])[0]
        result = "⚠️ High risk of heart disease" if prediction == 1 else "✅ Low risk of heart disease"
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

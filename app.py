
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model/heart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    try:
        inputs = [float(request.form[feature]) for feature in features]
        prediction = model.predict([inputs])[0]
        result = 'High Risk of Heart Disease' if prediction == 1 else 'Low Risk of Heart Disease'
    except:
        result = "Invalid input. Please enter numeric values only."
    return render_template('result.html', result=result)

# Updated for Render deployment
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
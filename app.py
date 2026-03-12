from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['pregnancies'],
        data['glucose'],
        data['bloodpressure'],
        data['skinthickness'],
        data['insulin'],
        data['bmi'],
        data['dpf'],
        data['age'],
    ]
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction  = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return jsonify({
        'prediction': int(prediction),
        'risk_score': round(float(probability) * 100, 1),
        'result':     'Diabetic' if prediction == 1 else 'Non-Diabetic'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
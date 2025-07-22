from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Dummy training for demonstration (You can replace this with a real trained model)
# Features: [hours, attendance, assignments, previous]
X = np.array([
    [2, 70, 3, 60],
    [4, 85, 4, 65],
    [5, 90, 5, 70],
    [6, 95, 6, 75],
    [7, 98, 7, 80],
    [8, 100, 8, 85],
])
y = np.array([50, 60, 70, 80, 90, 95])

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        hours = float(data['hours'])
        attendance = float(data['attendance'])
        assignments = int(data['assignments'])
        previous = float(data['previous'])

        input_features = np.array([[hours, attendance, assignments, previous]])
        predicted_score = model.predict(input_features)[0]

        return jsonify({'predicted_score': round(predicted_score, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

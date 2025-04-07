from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('..\\models\\ros_rf_model.pkl')
scaler = joblib.load('..\\models\\scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        input_data = {
            'Pregnancies': [data['Pregnancies']],
            'Glucose': [data['Glucose']],
            'BloodPressure': [data['BloodPressure']],
            'SkinThickness': [data['SkinThickness']],
            'Insulin': [data['Insulin']],
            'BMI': [data['BMI']],
            'DiabetesPedigreeFunction': [data['DiabetesPedigreeFunction']],
            'Age': [data['Age']]
        }

        df = pd.DataFrame(input_data)
        features = scaler.transform(df)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

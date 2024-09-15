from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from werkzeug.exceptions import BadRequest

app = Flask(__name__)



# Load the model and scaler
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    app.logger.error(f"Error loading model or scaler: {e}")
    exit(1)

# Define the expected columns based on your training data
EXPECTED_COLUMNS = ['area', 'bedrooms', 'bathrooms', 'stories',
                    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
                    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        data = request.form.to_dict()

        # Validate required fields
        required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'furnishingstatus']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Create a dictionary with all expected columns initialized to 0
        input_data = {col: 0 for col in EXPECTED_COLUMNS}

        # Update the input data with the values from the form
        for key, value in data.items():
            if key in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
                input_data[key] = 1 if value == 'on' else 0
            elif key == 'furnishingstatus':
                if value not in ['furnished', 'semi-furnished', 'unfurnished']:
                    raise ValueError(f"Invalid value for furnishingstatus: {value}")
                if value == 'semi-furnished':
                    input_data['furnishingstatus_semi-furnished'] = 1
                elif value == 'unfurnished':
                    input_data['furnishingstatus_unfurnished'] = 1
            else:
                try:
                    input_data[key] = float(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {key}: {value}. Expected a number.")

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Apply standard scaling
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df_scaled)[0]

        # Return the result
        return jsonify({'prediction': float(prediction)})

    except ValueError as e:
        app.logger.error(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({'error': "An unexpected error occurred. Please try again later."}), 500

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': "Invalid request. Please check your input and try again."}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"An error occurred: {str(e)}")
    return jsonify({'error': "An internal server error occurred. Please try again later."}), 500

if __name__ == '__main__':
    # Ensure Flask is listening on all interfaces (0.0.0.0) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)

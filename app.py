from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tempfile

# Define paths for model files on GitHub
MODEL_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/lstm_model.h5'
LABEL_ENCODER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/label_encoder.pkl'
TOKENIZER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/tokenizer.pkl'

app = Flask(__name__)

# Initialize variables for model and dependencies
model = None
label_encoder = None
tokenizer = None
MAX_SEQUENCE_LENGTH = 100

def load_model_and_dependencies():
    global model, label_encoder, tokenizer
    try:
        # Download and load model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            response = requests.get(MODEL_URL, allow_redirects=True)
            response.raise_for_status()  # Raise exception for non-200 status code
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        model = load_model(tmp_file_path)

        # Load label encoder
        response = requests.get(LABEL_ENCODER_URL, allow_redirects=True)
        response.raise_for_status()  # Raise exception for non-200 status code
        label_encoder = joblib.load(response.content)

        # Load tokenizer
        response = requests.get(TOKENIZER_URL, allow_redirects=True)
        response.raise_for_status()  # Raise exception for non-200 status code
        tokenizer = joblib.load(response.content)

        print("Model and dependencies loaded successfully.")
    except Exception as e:
        print("Error loading model and dependencies:", e)

# Load model and dependencies on startup
load_model_and_dependencies()

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from request
        user_input = request.json['message']
        
        # Validate input
        if not user_input:
            return jsonify({'error': 'Message is required'}), 400
        
        # Preprocess the input
        x_test = tokenizer.texts_to_sequences([user_input])
        x_test = np.array(x_test).squeeze()
        x_test = pad_sequences([x_test], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

        # Make prediction using the model
        y_pred = model.predict(x_test)
        predicted_label = label_encoder.inverse_transform([np.argmax(y_pred)])[0]

        return jsonify({'response': predicted_label})
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request'}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

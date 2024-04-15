from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer
import urllib.request

# Load the model, label encoder, and tokenizer
# model = load_model('D:/mha/lstm_model.h5')
# label_encoder = joblib.load('D:/mha/label_encoder.pkl')
# tokenizer = joblib.load('D:/mha/tokenizer.pkl')
# Define paths for model files on GitHub

MODEL_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/lstm_model.h5'
LABEL_ENCODER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/label_encoder.pkl'
TOKENIZER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/tokenizer.pkl'


# Load the model from the file
# model = load_model('lstm_model.h5')


# Load the model from the file
# model = load_model('label_encoder.pkl')
# Attempt to download the file
try:
    with urllib.request.urlopen(LABEL_ENCODER_URL) as response:
        # Load the label encoder
        label_encoder = joblib.load(response)
except Exception as e:
    print("Error:", e)
# label_encoder = joblib.load(LABEL_ENCODER_URL)
# Attempt to download the file
try:
    with urllib.request.urlopen(MODEL_URL) as response:
        # Load the LSTM model
        model = joblib.load(response)
except Exception as e:
    print("Error:", e)
# model = joblib.load(MODEL_URL)
tokenizer = joblib.load(TOKENIZER_URL)

# Load the model from the file


# Load the model, label encoder, and tokenizer from GitHub
# model = load_model(requests.get(MODEL_URL, allow_redirects=True))
# label_encoder = joblib.load(requests.get(LABEL_ENCODER_URL, allow_redirects=True).content)
# tokenizer = joblib.load(requests.get(TOKENIZER_URL, allow_redirects=True).content)

app = Flask(__name__)
MAX_SEQUENCE_LENGTH = 100
# Define route for processing user input
@app.route('/')
def hello():
    return 'Hello, World!'
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from request
    user_input = request.json['message']

    # Preprocess the input
    x_test = tokenizer.texts_to_sequences([user_input])
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    # Make prediction using the model
    y_pred = model.predict(x_test)
    predicted_label = label_encoder.inverse_transform([np.argmax(y_pred)])[0]

    return jsonify({'response': predicted_label})



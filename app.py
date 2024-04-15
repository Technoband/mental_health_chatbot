from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
# from keras.layers import TFSMLayer
from tensorflow.keras.layers import TFSMLayer


# Load the model, label encoder, and tokenizer
# model = load_model('D:/mha/lstm_model.h5')
# label_encoder = joblib.load('D:/mha/label_encoder.pkl')
# tokenizer = joblib.load('D:/mha/tokenizer.pkl')
# Define paths for model files on GitHub

MODEL_URL = './models/lstm_model.h5'
LABEL_ENCODER_URL = './models/label_encoder.pkl'
TOKENIZER_URL = './models/tokenizer.pkl'

response = requests.get(LABEL_ENCODER_URL)

# Load the label encoder
label_encoder = joblib.load(response.content)

# Assuming `data` is your input data that needs to be encoded
# encoded_data = label_encoder.transform(data)

# Load the model from the file
model = load_model(MODEL_URL)

# Load the model from the file
label_encoder = load_model(LABEL_ENCODER_URL)

# Load the model from the file
tokenizer= load_model(TOKENIZER_URL)



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



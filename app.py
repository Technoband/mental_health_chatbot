from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from io import BytesIO

# Load the model, label encoder, and tokenizer
# model = load_model('D:/mha/lstm_model.h5')
# label_encoder = joblib.load('D:/mha/label_encoder.pkl')
# tokenizer = joblib.load('D:/mha/tokenizer.pkl')
# Define paths for model files on GitHub
MODEL_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/lstm_model.h5'
LABEL_ENCODER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/label_encoder.pkl'
TOKENIZER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/tokenizer.pkl'

response = requests.get(MODEL_URL)
response.raise_for_status()
# Load the model from the response content
# model = load_model(BytesIO(response.content))
# Save the content to a file
with open('lstm_model.h5', 'wb') as f:
    f.write(response.content)

# Load the model from the file
model = load_model('lstm_model.h5')

# Download and load label encoder
response_label_encoder = requests.get(LABEL_ENCODER_URL)
response_label_encoder.raise_for_status()
# label_encoder = joblib.load(BytesIO(response_label_encoder.content))
# Save the content to a file
with open('label_encoder.pkl', 'wb') as f:
    f.write(response.content)

# Load the model from the file
model = load_model('label_encoder.pkl')

# Download and load tokenizer
response_tokenizer = requests.get(TOKENIZER_URL)
response_tokenizer.raise_for_status()
# tokenizer = joblib.load(BytesIO(response_tokenizer.content))
with open('tokenizer.pkl', 'wb') as f:
    f.write(response.content)

# Load the model from the file
model = load_model('tokenizer.pkl')

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



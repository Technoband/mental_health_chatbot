from flask import Flask, request, jsonify
import joblib
import numpy as np
import random
import re
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tempfile
import pandas as pd
import io

app = Flask(__name__)

# Define URLs for resources
MODEL_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/lstm_model.h5'
TOKENIZER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/tokenizer.pkl'
LABEL_ENCODER_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/label_encoder.pkl'
DF_EXPANDED_URL = 'https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/df_expanded.csv'

# Define maximum sequence length
MAX_SEQUENCE_LENGTH = 18

# Load model and dependencies
def load_resources():
    global tokenizer, label_encoder, df_expanded, model
    # Download and load model
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        tmp_file.write(response.content)
        model = load_model(tmp_file.name)
    # Load the tokenizer from URL and save it to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        response = requests.get(TOKENIZER_URL)
        response.raise_for_status()
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    # Load the tokenizer from the temporary file
    tokenizer = joblib.load(tmp_file_path)

    # Download and load label encoder
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        response = requests.get(LABEL_ENCODER_URL)
        response.raise_for_status()
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    # Load the label encoder from the temporary file
    label_encoder = joblib.load(tmp_file_path)

    # Download and load expanded responses data
    response = requests.get(DF_EXPANDED_URL)
    response.raise_for_status()
    df_expanded = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# Initialize resources
load_resources()
global negative_words 
negative_words = ["no", "not", "don't", "can't", "didn't", "won't", "stressed", "depressed", "depress", "stressful", "stress", "anxiety", "nothing", "nobody", "none", "neither", "nor"]
# Function to analyze input for negative words
def analyze_input(input_text):
    return any(word in input_text.lower() for word in negative_words)

# Function to generate answer
def generate_answer(pattern, negative_count):
    # Preprocess the user input pattern
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    # Tokenize the input pattern
    x_test = tokenizer.texts_to_sequences(text)
    if not x_test:  # Check if x_test is empty
        return negative_count, None  # Return without processing if input is empty

    # Pad the sequences only if x_test contains sequences
    if isinstance(x_test[0], list):
        x_test = pad_sequences(x_test, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    else:
        # Handle case where x_test contains a single sequence
        x_test = pad_sequences([x_test], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    # Predict the intent using the model
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = label_encoder.inverse_transform([y_pred])[0]

    # Retrieve responses associated with the predicted intent
    responses = df_expanded[df_expanded['tag'] == tag]['responses'].values[0]

    # negative_words = ["no", "not", "don't", "can't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "never", "nowhere", "nothing", "nobody", "none", "neither", "nor"]
    # Check if the response is negative
    is_negative = any(response.lower() in pattern.lower() for response in negative_words)

    if is_negative:
        negative_count += 1
    else:
        negative_count = 0

    # If there have been 4-5 consecutive negative messages, suggest doctor help
    if negative_count >= 1:
        response = "It seems like you might need professional help. Would you like me to provide you with the contact information of a doctor?"
        return negative_count, response

    # Randomly choose a response
    response = random.choice(responses)
    return negative_count, response

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data.get('message')
    negative_count = data.get('negative_count', 0)

    if not user_input:
        return jsonify({'error': 'Message is required'}), 400

    negative_count, response = generate_answer(user_input, negative_count)

    return jsonify({'response': response, 'negative_count': negative_count})

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0')

from flask import Flask, request, jsonify
import joblib
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Load the model, label encoder, and tokenizer
# model = joblib.load('/content/drive/MyDrive/mha/model.h5')
model = load_model('D:/mha/lstm_model.h5')
label_encoder = joblib.load('D:/mha/label_encoder.pkl')
tokenizer = joblib.load('D:/mha/tokenizer.pkl')

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

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import numpy as np
import random
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and dependencies
model = load_model('https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/lstm_model.h5')  # Update with the path to your saved model
tokenizer = joblib.load('https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/tokenizer.pkl')  # Update with the path to your saved tokenizer
label_encoder = joblib.load('https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/label_encoder.pkl')  # Update with the path to your saved label encoder
df_expanded = joblib.load('https://raw.githubusercontent.com/Technoband/mental_health_chatbot/main/models/df_expanded.csv')
# Load expanded responses data (df_expanded)
# You may need to reload your data if it's not available in the current environment

# Define maximum sequence length
MAX_SEQUENCE_LENGTH = 100

def generate_answer(pattern, negative_count):
    # Preprocess the user input pattern
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    # Tokenize and pad the input pattern
    x_test = tokenizer.texts_to_sequences(text)
    if not x_test:  # Check if x_test is empty
        return negative_count, None  # Return without processing if input is empty

    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    # Predict the intent using the model
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = label_encoder.inverse_transform([y_pred])[0]

    # Retrieve responses associated with the predicted intent
    responses = df_expanded[df_expanded['tag'] == tag]['responses'].values[0]

    negative_words = ["no", "not", "don't", "can't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "never", "nowhere", "nothing", "nobody", "none", "neither", "nor"]
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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

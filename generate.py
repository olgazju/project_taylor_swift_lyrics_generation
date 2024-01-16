from fastapi import FastAPI, HTTPException, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from fastapi.encoders import jsonable_encoder
import numpy as np
import pickle

app = FastAPI()

# Load the model
model = load_model('model/my_lyric_model.h5')

# Load max_sequence_len in your FastAPI service
with open('model/max_sequence_len.txt', 'r') as f:
    max_sequence_len = int(f.read())

# Load the tokenizer
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def sample_with_temperature(probabilities, temperature=1.0):
    # Adjust the probabilities with temperature
    probabilities = np.asarray(probabilities).astype('float64')
    probabilities = np.log(probabilities + 1e-10) / temperature  # Adding a small constant to avoid division by zero
    exp_probs = np.exp(probabilities)
    probabilities = exp_probs / np.sum(exp_probs)

    # Sample the next word based on the adjusted probabilities
    choices = range(len(probabilities))  # This should be the range of your vocabulary
    next_word = np.random.choice(choices, p=probabilities)

    return next_word

def generate_with_temperature(model, seed_text, temperature=1.0):

    next_words = 100
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        probabilities = model.predict(token_list, verbose=0)[0]  # Get softmax probabilities

        # Use temperature to adjust the probabilities and sample the next word
        predicted = sample_with_temperature(probabilities, temperature=temperature)  # Adjust the temperature as needed

        # Convert the predicted token to a word and update the seed text
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    return seed_text

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    print("data", data)

    # Generate text using the model
    result = generate_with_temperature(model, data['prompt'], float(data['temp']))
    return jsonable_encoder({"result": result})

# Run the server: uvicorn generate:app --reload

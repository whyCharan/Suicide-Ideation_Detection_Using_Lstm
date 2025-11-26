import sys
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add parent directory to path to import src
sys.path.append(os.path.abspath('..'))
from src.preprocess import clean_text

# Paths
MODEL_PATH = '../models/lstm_suicide_model.h5'
TOKENIZER_PATH = '../models/tokenizer.pkl'
LE_PATH = '../models/label_encoder.pkl'

def load_assets():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Loading label encoder...")
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le

def predict(text, model, tokenizer, le):
    MAX_LEN = 100
    cleaned_text = clean_text(text)
    print(f"Original: '{text}'")
    print(f"Cleaned: '{cleaned_text}'")
    
    if not cleaned_text:
        print("Warning: Cleaned text is empty.")
        # Handle empty text case if needed, e.g., return default or skip
    
    seq = tokenizer.texts_to_sequences([cleaned_text])
    # print(f"Sequence: {seq}")
    
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    
    prob = model.predict(pad, verbose=0)[0][0]
    # Adjust threshold if needed, but 0.5 is standard for balanced binary
    label_idx = int(prob > 0.5) 
    label = le.inverse_transform([label_idx])[0]
    
    print(f"Probability: {prob:.4f}")
    print(f"Prediction: {label}")
    print("-" * 30)

if __name__ == "__main__":
    try:
        model, tokenizer, le = load_assets()
        
        test_cases = [
            "life is not worth living anymore",
            "Life is so much good and i am so happy in this life",
            "Life is beautiful and I am grateful for everything I have.",
            "I want to kill myself",
            "I love my family",
            "I am feeling a bit sad but I will be okay",
            "" # Empty string test
        ]
        
        print("\n--- Running Predictions ---\n")
        for text in test_cases:
            predict(text, model, tokenizer, le)
            
    except Exception as e:
        print(f"Error: {e}")

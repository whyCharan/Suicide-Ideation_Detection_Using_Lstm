import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath('..'))
from src.preprocess import clean_text

# Page config
st.set_page_config(
    page_title="Suicide Ideation Detection",
    page_icon="🧠",
    layout="centered"
)

# Load assets
@st.cache_resource
def load_assets():
    model_path = '../models/lstm_suicide_model.h5'
    tokenizer_path = '../models/tokenizer.pkl'
    le_path = '../models/label_encoder.pkl'
    
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le

try:
    model, tokenizer, le = load_assets()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# UI
st.title("Suicide Ideation Detection")
st.write("Enter text below to analyze its sentiment regarding suicide ideation.")

text_input = st.text_area("Input Text", height=150)

if st.button("Analyze"):
    if text_input:
        cleaned_text = clean_text(text_input)
        
        if not cleaned_text:
            st.warning("Input text contains no valid words after cleaning.")
        else:
            # Preprocess
            MAX_LEN = 100
            seq = tokenizer.texts_to_sequences([cleaned_text])
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            
            # Predict
            prob = model.predict(pad, verbose=0)[0][0]
            label_idx = int(prob > 0.5)
            label = le.inverse_transform([label_idx])[0]
            
            # Display result
            st.subheader("Result")
            if label == 'suicide':
                st.error(f"Prediction: {label.upper()}")
            else:
                st.success(f"Prediction: {label.upper()}")
            
            st.write(f"Confidence (Suicide Probability): {prob:.4f}")
            
            with st.expander("Debug Info"):
                st.write(f"Cleaned Text: {cleaned_text}")
                st.write(f"Token Sequence: {seq}")
    else:
        st.warning("Please enter some text.")

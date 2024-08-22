import streamlit as st
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import generate_name

MODEL_PATH = "outputs/train_run_2024-08-22_21-09-19/best.pth"
TOYNAMER_MODEL = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: white;'>ToyNamer - Have Fun!</h1>", unsafe_allow_html=True)

    # Apply custom CSS to style the button and center it
    st.markdown(
        """
        <style>
        .stButton button {
            font-size: 24px;
            padding: 16px 24px;
            background-color: #417499;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            display: block;
            margin: 0 auto;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
        .generated-name {
            font-size: 36px;
            color: #fff4f2;
            text-align: center;
            margin-top: 30px;
        }        
        </style>
        """,
        unsafe_allow_html=True
    )

    # Temperature slider
    temperature = st.slider('Temperature', 0.25, 1.25, 0.75)

    # Centered large button
    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust the column proportions
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add a line break for spacing
        if st.button('Generate Name'):
            name = generate_name(model=TOYNAMER_MODEL, temperature=temperature)
            st.markdown(f'<div class="generated-name">{name}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

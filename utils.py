import base64
import cv2
import json
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image


def detect(img, model, class_names):
    img_resized = cv2.resize(img, (256, 256))
    img_resized = img_resized.astype(np.float32)
    img_processed = np.expand_dims(img_resized, axis=0)
    img_processed /= 255.0
    predictions = model.predict(img_processed)
    score = predictions[0][0]
    if score < 0.5 : 
        confidence_score = (1-score)*100
        prediction = class_names['0']
    else :
        confidence_score = score*100
        prediction = class_names['1']
    confidence_score = f"{confidence_score:.2f} %"
    results = {'detection': prediction, 'score': confidence_score}  
    return results


def load_class_names(file_path):
    with open(file_path, 'r') as json_file:
        class_names = json.load(json_file)
    return class_names


def load_model_keras(model_path):
    model = load_model(model_path)
    return model


def set_background(img_file):
    with open(img_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

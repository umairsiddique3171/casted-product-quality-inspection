import streamlit as st
import cv2
import pickle
import json
import numpy as np
from utils import detect, load_class_names, load_model , set_background

import warnings
warnings.filterwarnings("ignore")


# set background
set_background('background_img.jpg')

# set title
st.title("Casted Product Quality Inspection")

# set header
st.header("Please upload an image")

# set uploader
file = st.file_uploader('',type=['png','jpg','jpeg'])

# load classifier
model_path = r"C:\Users\US593\OneDrive\Desktop\casted-product-quality-inspection\model_training\model.h5"
model = load_model(model_path)

# load class_names
file_path = r"C:\Users\US593\OneDrive\Desktop\casted-product-quality-inspection\class_dict.json"
class_names = load_class_names(file_path)

# display image
if file is not None: 
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img,use_column_width=True)

# classify image
    results = detect(img,model,class_names)
    detection = results['detection']
    confidence_score = results['score']

# show results
    st.write(f"### Detection : {detection}")
    st.write(f"#### Confidence Score : {confidence_score}")
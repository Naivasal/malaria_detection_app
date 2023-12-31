import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import cv2


# Title of the app
st.title("Malaria Detection App")
model = load_model('malariadetection_modle.keras')
# File uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
image_size = (33, 33)
if uploaded_file is not None:
    m_image = Image.open(uploaded_file).resize(image_size)
    prediction = model.predict(np.expand_dims(m_image,axis=0))
    #st.text("prediction : :" + str(prediction))
    prediction = prediction[0][0]
    if prediction > 0.5:
        st.header("Malaria Detected")
        st.subheader("The uploaded image suggest the presence of Malaria")
        st.write("** please seek immediate attention from the professional")
    else:
        st.header("Malaria Not Detected")
        st.subheader("The uploaded image suggest that there is no sign of Malaria")
        st.write("** if you have any concern or symptoms , please seek help from the professional")

    # Display the original image
    st.image(m_image, caption="Original Image", use_column_width=True)

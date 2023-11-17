import streamlit as st
import cv2
import numpy as np
st.title("Testing Streamlit camera")
img = st.camera_input("Camera")

if img:
    st.image(img)
    img_array = np.asarray(bytearray(img.read()), dtype=np.uint8)
    n = cv2.imdecode(img_array, 1)
    minRange = np.array([0, 133, 77], np.uint8)
    maxRange = np.array([235, 173, 127], np.uint8)
    YCRn = cv2.cvtColor(n, cv2.COLOR_BGR2YCR_CB)
    skinArea = cv2.inRange(YCRn, minRange, maxRange)
    detectedSkin = cv2.bitwise_and(n, n, mask=skinArea)
    cv2.imshow('sin', detectedSkin)
    cv2.waitKey(0)
    edges = cv2.Canny(detectedSkin, 100, 200)
    cv2.imshow('sin', edges)

    # Append the image to the list

    # Display the image using Streamlit
    st.image(n, caption='Uploaded Image', use_column_width=True)
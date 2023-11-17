import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import os
 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model = keras.models.load_model("sign_recognition_model.h5")
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
               'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
               'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

# Reverse mapping from numerical labels to letters
reverse_labels_dict = {v: k for k, v in labels_dict.items()}
class VideoTransformer(VideoTransformerBase):
    image_counter = 0
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        process_image = image.copy()
        process_image.flags.writeable = False
        
        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        process_results = hands.process(process_image)
        #return img
        # minRange = np.array([0, 133, 77], np.uint8)
        # maxRange = np.array([235, 173, 127], np.uint8)
        # YCRn = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        # skinArea = cv2.inRange(YCRn, minRange, maxRange)
        # detectedSkin = cv2.bitwise_and(img, img, mask=skinArea)
        # # return detectedSkin
        # edges = cv2.Canny(detectedSkin, 100, 200)
        # return edges
        blank_image = np.zeros_like(image)
        if process_results.multi_hand_landmarks:
            for hand_landmarks in process_results.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                    blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        VideoTransformer.image_counter += 1
        # image_path = f"hand_image_{ VideoTransformer.image_counter}.png"
        # cv2.imwrite(image_path, blank_image)
        process_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY) 
        # Assuming images are grayscale
        
        process_image = cv2.resize(process_image, (64, 64)) 
        # image_path_1 = f"final_hand_image_{ VideoTransformer.image_counter}.png"
        # cv2.imwrite(image_path_1, process_image)
        process_image = process_image.astype("float32") / 255.0
        process_image = np.expand_dims(process_image, axis=0)
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        prediction = model.predict(process_image)
        # # # Decode the prediction to get the predicted character
        predicted_class = np.argmax(prediction)
        predicted_letter = reverse_labels_dict[predicted_class]
        # # print(prediction)
        image.flags.writeable = False
        results = hands.process(image)
        
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # predicted_class = "Hello"
        cv2.putText(image, f'Predicted Alphabet: {predicted_letter}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image  

st.title('ISL Detector')

webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

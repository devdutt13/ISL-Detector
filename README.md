# ISL-Detector
Indian Sign Language Detection using Mediapipe and CNN

The Indian Sign Language Recognition project aims to detect Indian Sign Language gestures from a live video feed. This is achieved by utilizing the following components:

- **Mediapipe:** A library for hand tracking and pose recognition. It is used to detect the position of hands and fingers in the captured video frames.

- **CNN Model:** A Convolutional Neural Network (CNN) model has been trained for sign language recognition. The hand and finger positions detected by Mediapipe are fed into this model for prediction.

- **Streamlit:** The user interface of the application is built using Streamlit, providing a simple and interactive way for users to interact with the Indian Sign Language recognition system.

## Features

- Real-time hand and finger tracking using Mediapipe.
- CNN model for  sign language prediction.
- User-friendly UI powered by Streamlit.

# Run the Application

To run the Indian Sign Language Recognition application, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/devdutt13/ISL-Detector.git

# 2. Navigate to the project directory
cd indian-sign-language-recognition

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py


# Emotion-detector
An end-to-end Computer Vision web application built with **PyTorch**, **ResNet34**, and **Gradio**. The system classifies facial emotion categories in real time using either uploaded images or live webcam feeds.

*Specifically Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.*

## Key Features
* **Deep Learning** using **ResNet34**
* **Interactive web interface** - allows real-time webcam capture, image uploading and shows top 3 predictions
* **Checkpoints** after each epoch to prevent overfitting

## Tech stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch, torchvision, timm
* **Web UI:** Gradio
* **Data Processors:** PIL, JSON, kagglehub

## How to use
1. Run the model trainer first with the downloaded database (my model used one downloaded from kagglehub with 6 classes)
2. Run the app
3. Navigate to the website with this link - http://127.0.0.1:7860


<img width="947" height="223" alt="image" src="https://github.com/user-attachments/assets/084572f2-b4c6-460e-a2c9-a1cadb01a2f9" />


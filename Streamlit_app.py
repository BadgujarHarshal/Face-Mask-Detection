# Streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("face_mask_detector_savedmodel")

CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

st.title("Face Mask Detection App")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Model returns: {'bbox': [...], 'class_output': [...]}
    predictions = model.predict(img_input)
    bbox = predictions['bbox'][0]
    cls = predictions['class_output'][0]

    x1, y1, x2, y2 = bbox
    pred_class = np.argmax(cls)

    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, CLASSES[pred_class], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(img, channels="BGR", caption=f"Prediction: {CLASSES[pred_class]}")

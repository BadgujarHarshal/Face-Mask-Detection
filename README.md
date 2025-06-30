# Face-Mask-Detection

Implemention of face mask detector using a CNN-based detection model.
Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
Contains images of people
Includes:- Classes: with_mask, without_mask, mask_weared_incorrect
  Assignment Tasks:
Load and parse annotation files (Pascal VOC XML → TensorFlow format)
Resize images uniformly (e.g., 224x224 or 416x416)
Encode bounding boxes and class labels
Split into train/val/test

Choose ONE of the following implementation paths:

Use pre-trained as feature extractor
Add custom regression head for
Add classification head for
Train using a : (MSE for boxes + Categorical CrossEntropy for class)

Modify SSD or YOLO-lite model for 2 classes
Use ,
Use tf.image.draw_bounding_boxes() to visualize predictions
Use model.fit() or custom training loop
Implement learning rate schedule (optional)
Apply : flip, rotate, color jitter
Report:
at IoU=0.5
per class
Overlay predicted bounding boxes on 10 random test images
Deploy as a using:
OpenCV + TensorFlow SavedModel
Streamlit-based web app
Quantize with (TFLite)

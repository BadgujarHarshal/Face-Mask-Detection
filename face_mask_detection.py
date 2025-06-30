!pip install kagglehub lxml tensorflow

import kagglehub
import os

dataset_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
print("Dataset path:", dataset_path)

import os, cv2, random
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import kagglehub
from google.colab import files

path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
image_dir = os.path.join(path, "images")
annotation_dir = os.path.join(path, "annotations")
CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
IMAGE_SIZE = 224

def load_data(image_dir, annotations_dir):
    data_by_class = defaultdict(list)
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"): continue
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        filename = root.find('filename').text
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        h, w = image.shape[:2]
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label not in CLASSES: continue
            box = obj.find('bndbox')
            x1 = int(box.find('xmin').text) / w
            y1 = int(box.find('ymin').text) / h
            x2 = int(box.find('xmax').text) / w
            y2 = int(box.find('ymax').text) / h
            class_id = CLASSES.index(label)
            data_by_class[class_id].append((image, [x1, y1, x2, y2], class_id))
    return data_by_class

def oversample(data_by_class, target_class, times):
    samples = data_by_class[target_class]
    data_by_class[target_class] = samples * times
    return data_by_class

data_by_class = load_data(image_dir, annotation_dir)
data_by_class = oversample(data_by_class, target_class=2, times=2)

all_data = []
for cls, samples in data_by_class.items():
    all_data.extend(samples)
random.shuffle(all_data)

images, bboxes, labels = [], [], []
for img, box, cls in all_data:
    images.append(img)
    bboxes.append(box)
    labels.append(tf.keras.utils.to_categorical(cls, num_classes=3))

images = np.array(images) / 255.0
bboxes = np.array(bboxes)
labels = np.array(labels)

X_train, X_test, y_train_box, y_test_box, y_train_label, y_test_label = train_test_split(
    images, bboxes, labels, test_size=0.2, random_state=42)

del images, bboxes, labels
import gc
gc.collect()

base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
x = GlobalAveragePooling2D()(base.output)
bbox = Dense(4, name='bbox')(x)
class_output = Dense(3, activation='softmax', name='class_output')(x)
model = Model(inputs=base.input, outputs=[bbox, class_output])
model.compile(
    optimizer='adam',
    loss={'bbox': 'mse', 'class_output': 'categorical_crossentropy'},
    metrics={'bbox': 'mse', 'class_output': 'accuracy'}
)

def data_generator(X, y_box, y_label, batch_size=4):
    while True:
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], {
                'bbox': y_box[i:i+batch_size],
                'class_output': y_label[i:i+batch_size]
            }

train_gen = data_generator(X_train, y_train_box, y_train_label)
val_gen = data_generator(X_test, y_test_box, y_test_label)

steps_per_epoch = len(X_train) // 4
val_steps = len(X_test) // 4

early_stop = EarlyStopping(monitor='val_class_output_accuracy', patience=3, mode='max', restore_best_weights=True)

model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=30,
    callbacks=[early_stop]
)

pred_bb, pred_cls = model.predict(X_test)
y_pred = np.argmax(pred_cls, axis=1)
y_true = np.argmax(y_test_label, axis=1)

print("\nClassification Report at IoU=0.5:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))

indices = random.sample(range(len(X_test)), 10)

for idx in indices:
    img = X_test[idx]
    pred_box = pred_bb[idx]
    pred_class = np.argmax(pred_cls[idx])

    if img.shape[:2] != (224, 224):
        print(f"Warning: Image shape is {img.shape[:2]} not (224, 224)")
        img = cv2.resize((img * 255).astype(np.uint8), (224, 224))
    else:
        img = (img * 255).astype(np.uint8)

    img_copy = img.copy()

    cv2.putText(img_copy, CLASSES[pred_class], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

model.save("face_mask_detector.h5")
model.save("face_mask_detector.keras")
model.export("face_mask_detector_savedmodel")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("face_mask_detector.tflite", "wb") as f:
    f.write(tflite_model)

files.download("face_mask_detector.h5")
files.download("face_mask_detector.keras")
!zip -r face_mask_detector_savedmodel.zip face_mask_detector_savedmodel
files.download("face_mask_detector_savedmodel.zip")
files.download("face_mask_detector.tflite")
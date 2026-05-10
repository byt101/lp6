import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from utils import (
    preprocess_image,
    extract_brain,
    segment_region,
    localize_tumor
)


# ---------------- SETTINGS ---------------- #

IMAGE_SIZE = 224

MODEL_PATH = "brain_tumor_model.keras"

IMAGE_PATH = "dataset/no/17 no.jpg"


# ---------------- LOAD MODEL ---------------- #

model = load_model(MODEL_PATH)


# ---------------- LOAD IMAGE ---------------- #

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found")
    exit()


# ---------------- CNN PREPROCESS ---------------- #

cnn_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

cnn_img = image.img_to_array(cnn_img)

cnn_img = cnn_img / 255.0

cnn_img = np.expand_dims(cnn_img, axis=0)


# ---------------- PREDICTION ---------------- #

prediction = model.predict(cnn_img, verbose=0)[0][0]


# ---------------- DECISION ---------------- #

THRESHOLD = 0.5

if prediction >= THRESHOLD:

    tumor_status = "Tumor Detected"

    confidence = round(prediction * 100, 2)

    tumor_present = True

else:

    tumor_status = "No Tumor Detected"

    confidence = round((1 - prediction) * 100, 2)

    tumor_present = False


# ---------------- IMAGE PROCESSING ---------------- #

original, gray, enhanced, blur = preprocess_image(img)

brain_mask = extract_brain(gray)

thresh, segmented = segment_region(
    blur,
    brain_mask
)


# ---------------- LOCALIZATION ---------------- #

if tumor_present:

    detected = localize_tumor(
        original,
        segmented
    )

else:

    detected = original.copy()

    cv2.putText(
        detected,
        "No Tumor Detected",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


# ---------------- SAVE OUTPUTS ---------------- #

os.makedirs("outputs", exist_ok=True)

cv2.imwrite("outputs/grayscale.jpg", gray)

cv2.imwrite("outputs/enhanced.jpg", enhanced)

cv2.imwrite("outputs/brain_mask.jpg", brain_mask)

cv2.imwrite("outputs/threshold.jpg", thresh)

cv2.imwrite("outputs/segmented.jpg", segmented)

cv2.imwrite("outputs/final_detection.jpg", detected)


# ---------------- TERMINAL OUTPUT ---------------- #

print("\n========== MRI ANALYSIS RESULT ==========\n")

print("Prediction Status   :", tumor_status)

print("Confidence Score    :", confidence, "%")

print("Outputs Saved In    : outputs/")

print("\n=========================================\n")


# ---------------- VISUALIZATION ---------------- #

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

detected_rgb = cv2.cvtColor(
    detected,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(original_rgb)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(brain_mask, cmap='gray')
plt.title("Brain Mask")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(thresh, cmap='gray')
plt.title("Threshold")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(segmented, cmap='gray')
plt.title("Segmented Region")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(detected_rgb)
plt.title(tumor_status)
plt.axis("off")

plt.tight_layout()

plt.show()
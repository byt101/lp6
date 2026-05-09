import cv2
import os
import matplotlib.pyplot as plt

from utils import (
    preprocess_image,
    segment_image,
    detect_tumor
)

# Input image
image_path = "dataset/yes/Y1.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

# Preprocessing
original, gray, enhanced, blur = preprocess_image(image)

# Segmentation
thresh, segmented = segment_image(blur)

# Detection
detected, status, area = detect_tumor(original, segmented)

# Convert to RGB
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
detected_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Save images
cv2.imwrite("outputs/grayscale.jpg", gray)
cv2.imwrite("outputs/enhanced.jpg", enhanced)
cv2.imwrite("outputs/threshold.jpg", thresh)
cv2.imwrite("outputs/segmented.jpg", segmented)
cv2.imwrite("outputs/final_detection.jpg", detected)

# Result
print("\n========== MRI ANALYSIS RESULT ==========")

if status:
    print("Tumor Status        : Tumor Detected")
else:
    print("Tumor Status        : No Tumor Detected")

print("Suspicious Area     :", int(area), "pixels")

print("Outputs Saved In    : outputs/")

print("=========================================\n")

# Visualization
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
plt.imshow(enhanced, cmap='gray')
plt.title("CLAHE Enhanced")
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
plt.title("Final Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
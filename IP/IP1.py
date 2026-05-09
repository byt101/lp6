import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image.jpg")

# Check image
if image is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Calculate Histograms
hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# Display Images
plt.figure(figsize=(12,8))

# Original Image
plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis("off")

# Original Histogram
plt.subplot(2,2,2)
plt.plot(hist_original)
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Equalized Image
plt.subplot(2,2,3)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

# Equalized Histogram
plt.subplot(2,2,4)
plt.plot(hist_equalized)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
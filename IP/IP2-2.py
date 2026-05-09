import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Intensity range
lower = 100
upper = 180

# Mask
mask = cv2.inRange(gray, lower, upper)

# Apply mask
result = cv2.bitwise_and(image, image, mask=mask)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Intensity Level Slicing")
plt.axis("off")

plt.tight_layout()
plt.show()
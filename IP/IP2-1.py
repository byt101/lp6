import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found")
    exit()

# Contrast Stretching Function
def contrast_stretch(channel):

    min_val = np.min(channel)
    max_val = np.max(channel)

    stretched = ((channel - min_val) /
                (max_val - min_val)) * 255

    return stretched.astype(np.uint8)

# Split channels
b, g, r = cv2.split(image)

# Apply stretching
b1 = contrast_stretch(b)
g1 = contrast_stretch(g)
r1 = contrast_stretch(r)

# Merge channels
stretched = cv2.merge([b1, g1, r1])

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB))
plt.title("Contrast Stretched Image")
plt.axis("off")

plt.tight_layout()
plt.show()
import cv2
import os

# Read image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found")
    exit()

# Original Size
original_size = os.path.getsize("image.jpg")

# JPEG Compression
cv2.imwrite(
    "compressed.jpg",
    image,
    [cv2.IMWRITE_JPEG_QUALITY, 50]
)

# PNG Compression
cv2.imwrite(
    "compressed.png",
    image,
    [cv2.IMWRITE_PNG_COMPRESSION, 9]
)

# WEBP Compression
cv2.imwrite(
    "compressed.webp",
    image,
    [cv2.IMWRITE_WEBP_QUALITY, 50]
)

# Sizes
jpg_size = os.path.getsize("compressed.jpg")
png_size = os.path.getsize("compressed.png")
webp_size = os.path.getsize("compressed.webp")

# Compression Ratios
jpg_cr = original_size / jpg_size
png_cr = original_size / png_size
webp_cr = original_size / webp_size

print("\nOriginal Size:", original_size, "bytes")

print("\nJPEG")
print("Compressed Size:", jpg_size)
print("Compression Ratio:", round(jpg_cr, 2))

print("\nPNG")
print("Compressed Size:", png_size)
print("Compression Ratio:", round(png_cr, 2))

print("\nWEBP")
print("Compressed Size:", webp_size)
print("Compression Ratio:", round(webp_cr, 2))
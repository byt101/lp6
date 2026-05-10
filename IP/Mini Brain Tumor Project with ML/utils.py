import cv2
import numpy as np


# ---------------- PREPROCESS IMAGE ---------------- #

def preprocess_image(image):

    image = cv2.resize(image, (512, 512))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    return image, gray, enhanced, blur


# ---------------- BRAIN MASK ---------------- #

def extract_brain(gray):

    _, mask = cv2.threshold(
        gray,
        20,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask


# ---------------- SEGMENTATION ---------------- #

def segment_region(blur, brain_mask):

    masked = cv2.bitwise_and(
        blur,
        blur,
        mask=brain_mask
    )

    _, thresh = cv2.threshold(
        masked,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    return thresh, closing


# ---------------- LOCALIZATION ---------------- #

def localize_tumor(image, segmented):

    contours, _ = cv2.findContours(
        segmented,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = image.copy()

    largest_contour = None
    largest_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > largest_area:
            largest_area = area
            largest_contour = cnt

    if largest_contour is not None and largest_area > 1200:

        x, y, w, h = cv2.boundingRect(largest_contour)

        cv2.rectangle(
            output,
            (x, y),
            (x+w, y+h),
            (0,0,255),
            3
        )

        cv2.putText(
            output,
            "Suspicious Region",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    return output
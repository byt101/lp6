import cv2
import numpy as np

# Preprocessing
def preprocess_image(image):

    image = cv2.resize(image, (512, 512))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    return image, gray, enhanced, blur


# Segmentation
def segment_image(blur):

    _, thresh = cv2.threshold(
        blur,
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
        iterations=2
    )

    return thresh, closing


# Tumor Detection
def detect_tumor(image, segmented):

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

    tumor_detected = False

    if largest_contour is not None and largest_area > 1500:

        tumor_detected = True

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
            "Tumor Detected",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    else:

        cv2.putText(
            output,
            "No Tumor Detected",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    return output, tumor_detected, largest_area
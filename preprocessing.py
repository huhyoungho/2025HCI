import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return processed
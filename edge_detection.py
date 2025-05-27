import cv2
import numpy as np

def apply_edge_detection(image):
    canny = cv2.Canny(image, 50, 200)
    """
    soble test 해볼 경우 해당 코드 사용하면 됨
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)
    """
    return canny

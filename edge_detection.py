import cv2
import numpy as np

def apply_edge_detection(image):
    # 이미지가 3채널이면 BGR->GRAY 변환, 아니면 그대로 사용
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    canny = cv2.Canny(gray, 100, 200)
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)
    """
    return canny

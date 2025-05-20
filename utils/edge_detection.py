import cv2

def sobel_edge(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

def canny_edge(img):
    return cv2.Canny(img, 100, 200)
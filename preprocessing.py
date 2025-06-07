import cv2
import numpy as np

def histogram_equalization(y_channel):
    hist, bins = np.histogram(y_channel.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * (255 / cdf[-1])
    equalized = np.interp(y_channel.flatten(), bins[:-1], cdf_normalized)
    return equalized.reshape(y_channel.shape).astype(np.uint8)

def histogram_stretching(y_channel):
    # 최소, 최대 밝기값 계산
    min_val = np.min(y_channel)
    max_val = np.max(y_channel)
    # 스트레칭 공식 적용
    stretched = (y_channel - min_val) * 255.0 / (max_val - min_val + 1e-8)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def rgb_to_y(image):
    b, g, r = cv2.split(image)
    rgb = cv2.merge((r, g, b))
    n = np.array([0.257, 0.504, 0.098])
    y = np.tensordot(rgb, n, axes=([-1], [0])) + 16
    return y

def color_preprocessing(image, s):
    b, g, r = cv2.split(image)
    Y = rgb_to_y(image)
    #Y_eq = histogram_equalization(Y)
    Y_eq = histogram_stretching(Y)
    # 밝기 계수 추가
    brightness_factor = 1.15  # 15% 더 밝게
    Y_eq = np.clip(Y_eq * brightness_factor, 0, 255).astype(np.uint8)
    R = np.clip(Y_eq * np.power((r / (Y + 1e-8)), s), 0, 255).astype(np.uint8)
    G = np.clip(Y_eq * np.power((g / (Y + 1e-8)), s), 0, 255).astype(np.uint8)
    B = np.clip(Y_eq * np.power((b / (Y + 1e-8)), s), 0, 255).astype(np.uint8)
    result = cv2.merge((B, G, R))
    return result

def preprocessing_image(image, s=0.5):
    # s 파라미터를 받아서 처리
    return color_preprocessing(image, s)


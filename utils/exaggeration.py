import cv2
import numpy as np

def scale_feature(img, points, scale_factor):
    center = np.mean(points, axis=0).astype(int)
    new_img = img.copy()

    for pt in points:
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        new_x = int(center[0] + dx * scale_factor)
        new_y = int(center[1] + dy * scale_factor)
        cv2.circle(new_img, (new_x, new_y), 2, (255, 0, 0), -1)

    return new_img
import cv2
import numpy as np

FACIAL_REGIONS = {
    "left_eye": [33, 133, 160, 159, 158, 157, 173, 246],
    "right_eye": [362, 263, 387, 386, 385, 384, 398, 466],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "nose": [1, 2, 98, 327, 168, 195],
    "left_cheek": [50, 101, 118, 123, 147, 213],
    "right_cheek": [280, 347, 330, 352, 376, 433],
    "chin": [152, 377, 400, 378, 379, 365, 397, 288],
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356],
    "jaw": [172, 136, 150, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288],
    # 필요시 추가 부위
}

SCALE_MAPPING = {
    "크게": 1.5,
    "작게": 0.7
}

def warp_region(image, src_points, scale=1.5):
    src_points = np.array(src_points, dtype=np.float32)
    center = np.mean(src_points, axis=0)
    translated = src_points - center
    scaled = translated * scale
    dst_points = scaled + center
    M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
    warped = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, np.int32(dst_points), (255, 255, 255))
    warped_region = cv2.bitwise_and(warped, mask)
    background_removed = cv2.bitwise_and(image, cv2.bitwise_not(mask))
    return cv2.add(warped_region, background_removed)

def apply_modification(image, landmarks, modifications):
    output = image.copy()
    for region_key, action in modifications.items():
        if region_key in FACIAL_REGIONS:
            indexes = FACIAL_REGIONS[region_key]
            scale = SCALE_MAPPING.get(action, 1.0)
            region_coords = np.array([landmarks[i] for i in indexes], dtype=np.int32)
            output = warp_region(output, region_coords, scale=scale)
    return output
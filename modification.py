import cv2
import numpy as np

FACIAL_REGIONS = {
    "left_eye": [33, 133, 160, 159, 158, 157, 173, 246],
    "right_eye": [362, 263, 387, 386, 385, 384, 398, 466],
    # 입 전체(윗입술+아랫입술) landmark index로 확장
    "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    # 코 인덱스 수정
    "nose": [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48,
              278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274],
    "chin": [152, 377, 400, 378, 379, 365, 397, 288],
    "left_cheek": [50, 101, 118, 123, 147, 213],
    "right_cheek": [280, 347, 330, 352, 376, 433]
}

SCALE_MAPPING = {
    "크게": 1.0,
    "작게": 0.7
}

# 부위별 기본 scale 값 지정
REGION_SCALE = {
    "left_eye": 1.5,
    "right_eye": 1.5,
    "mouth": 1.2,
    "nose": 1.1,
    "chin": 1.0,
    "left_cheek": 1.05,
    "right_cheek": 1.05
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
        # 볼 입력 시 양쪽 볼 모두 적용
        if region_key == "볼":
            for cheek in ["left_cheek", "right_cheek"]:
                indexes = FACIAL_REGIONS[cheek]
                base_scale = REGION_SCALE.get(cheek, 1.0)
                action_scale = SCALE_MAPPING.get(action, 1.0)
                scale = base_scale * action_scale
                region_coords = np.array([landmarks[i] for i in indexes], dtype=np.int32)
                output = warp_region(output, region_coords, scale=scale)
            continue
        # 입력 제한: 눈, 코, 입, 턱만 허용
        if region_key not in FACIAL_REGIONS:
            continue
        indexes = FACIAL_REGIONS[region_key]
        base_scale = REGION_SCALE.get(region_key, 1.0)
        action_scale = SCALE_MAPPING.get(action, 1.0)
        scale = base_scale * action_scale
        region_coords = np.array([landmarks[i] for i in indexes], dtype=np.int32)
        output = warp_region(output, region_coords, scale=scale)
    return output
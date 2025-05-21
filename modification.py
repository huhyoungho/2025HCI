import cv2
import numpy as np

# Mediapipe 기준 부위 별 인덱스 (대표 포인트 예시)
FACIAL_PARTS = {
    "눈": list(range(33, 133)),
    "코": list(range(1, 20)),
    "입": list(range(61, 88)),
    "귀": list(range(234, 267)) + list(range(454, 481)),
    "광대": list(range(93, 103)) + list(range(323, 333)),
    "이마": list(range(10, 20)),
}

# 확대/축소 비율
SCALE_MAP = {
    "크게": 1.3,
    "작게": 0.7,
}

def apply_modification(img, landmarks, modifications):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = gray.copy()

    h, w = gray.shape

    for part, scale_type in modifications.items():
        if part not in FACIAL_PARTS:
            continue

        indices = FACIAL_PARTS[part]
        points = [landmarks[i] for i in indices if i < len(landmarks)]

        if not points:
            continue

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = max(min(x_coords), 0), min(max(x_coords), w)
        y_min, y_max = max(min(y_coords), 0), min(max(y_coords), h)

        roi = result[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        scale = SCALE_MAP.get(scale_type, 1.0)
        new_w = int((x_max - x_min) * scale)
        new_h = int((y_max - y_min) * scale)

        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 중심 기준으로 위치 계산
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        x1 = max(center_x - new_w // 2, 0)
        y1 = max(center_y - new_h // 2, 0)
        x2 = min(x1 + new_w, w)
        y2 = min(y1 + new_h, h)

        resized = resized[:y2 - y1, :x2 - x1]  # 오버플로 방지
        result[y1:y2, x1:x2] = resized

    return result

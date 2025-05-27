import cv2
import numpy as np

FACIAL_REGIONS = {
    "left_eye": list(range(33, 42)),
    "right_eye": list(range(263, 272)),
    "nose": list(range(1, 6)) + list(range(168, 176)),
    "mouth": list(range(78, 88)) + list(range(308, 318)),
    "left_cheek": [50, 101, 118, 123],
    "right_cheek": [280, 347, 330, 352],
}

SCALE_MAPPING = {
    "크게": 2.0,
    "작게": 0.2
}

def apply_modification(image, landmarks, modifications):
    output = image.copy()

    for region_kr, action in modifications.items():
        if region_kr == "눈":
            indices = FACIAL_REGIONS["left_eye"] + FACIAL_REGIONS["right_eye"]
        elif region_kr == "코":
            indices = FACIAL_REGIONS["nose"]
        elif region_kr == "입":
            indices = FACIAL_REGIONS["mouth"]
        elif region_kr == "광대":
            indices = FACIAL_REGIONS["left_cheek"] + FACIAL_REGIONS["right_cheek"]
        else:
            continue

        scale = SCALE_MAPPING.get(action, 1.0)
        region_points = np.array([landmarks[i] for i in indices], dtype=np.int32)

        # 코 부위일 때 좌표 중심 기준 확대 적용
        if region_kr == "코":
            # 코 중심 계산 (평균)
            center = np.mean(region_points, axis=0)

            # 중심 기준 스케일링
            scaled_points = ((region_points - center) * scale + center).astype(np.int32)

            # 확대된 영역 bounding box
            x, y, w, h = cv2.boundingRect(scaled_points)
            if w == 0 or h == 0:
                continue

            # 원본 코 영역 bounding box (확대 이전)
            x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(region_points)
            roi = image[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig].copy()

            # 확대 후 영역만큼 크기 조정
            resized_roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)

            # 다각형 마스크 생성 (확대된 코 윤곽)
            mask = np.zeros((h, w), dtype=np.uint8)
            # scaled_points 좌표를 bounding box 기준으로 이동
            poly_points = scaled_points - [x, y]
            cv2.fillPoly(mask, [poly_points], 255)

            # 붙일 위치 계산 (이미지 경계 고려)
            paste_x = max(0, x)
            paste_y = max(0, y)
            paste_x2 = min(output.shape[1], paste_x + w)
            paste_y2 = min(output.shape[0], paste_y + h)

            # 크롭 사이즈 재조정
            crop_w = paste_x2 - paste_x
            crop_h = paste_y2 - paste_y
            resized_roi_cropped = resized_roi[0:crop_h, 0:crop_w]
            mask_cropped = mask[0:crop_h, 0:crop_w]

            if crop_w == 0 or crop_h == 0:
                continue

            roi_area = output[paste_y:paste_y2, paste_x:paste_x2]

            center_clone = (crop_w // 2, crop_h // 2)
            try:
                output[paste_y:paste_y2, paste_x:paste_x2] = cv2.seamlessClone(
                    resized_roi_cropped, roi_area, mask_cropped, center_clone, cv2.NORMAL_CLONE
                )
            except cv2.error as e:
                print(f"seamlessClone error: {e}")
                output[paste_y:paste_y2, paste_x:paste_x2] = resized_roi_cropped

        else:
            # 기존 방식 유지 (사각형 영역 리사이즈 후 붙이기)
            x, y, w, h = cv2.boundingRect(region_points)
            if w == 0 or h == 0:
                continue

            roi = image[y:y+h, x:x+w].copy()
            scaled_w = max(1, int(w * scale))
            scaled_h = max(1, int(h * scale))
            resized_roi = cv2.resize(roi, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            mask = 255 * np.ones(resized_roi.shape[:2], dtype=np.uint8)

            center_x = x + w // 2
            center_y = y + h // 2
            paste_x = max(0, center_x - scaled_w // 2)
            paste_y = max(0, center_y - scaled_h // 2)
            paste_x2 = min(output.shape[1], paste_x + scaled_w)
            paste_y2 = min(output.shape[0], paste_y + scaled_h)

            crop_w = paste_x2 - paste_x
            crop_h = paste_y2 - paste_y
            resized_roi_cropped = resized_roi[0:crop_h, 0:crop_w]
            mask_cropped = mask[0:crop_h, 0:crop_w]

            if crop_w == 0 or crop_h == 0:
                continue

            roi_area = output[paste_y:paste_y2, paste_x:paste_x2]
            center_clone = (crop_w // 2, crop_h // 2)
            try:
                output[paste_y:paste_y2, paste_x:paste_x2] = cv2.seamlessClone(
                    resized_roi_cropped, roi_area, mask_cropped, center_clone, cv2.NORMAL_CLONE
                )
            except cv2.error as e:
                print(f"seamlessClone error: {e}")
                output[paste_y:paste_y2, paste_x:paste_x2] = resized_roi_cropped
    return output
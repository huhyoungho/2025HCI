import cv2
import numpy as np

FACIAL_REGIONS = {
    "left_eye": [33, 133, 160, 159, 158, 157, 173, 246, 23, 24, 110],
    "right_eye": [362, 263, 387, 386, 385, 384, 398, 466, 443, 444, 276],
    "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "nose": [98, 327, 2, 195, 5, 4, 278, 279, 309, 456, 419, 248, 281],
    "chin": [152, 148, 176, 149, 150, 136, 172, 397, 365, 288],
    "left_cheek": [50, 101, 118, 123, 147, 213, 205, 206],
    "right_cheek": [280, 347, 330, 352, 376, 433, 426, 436]
}

SCALE_MAPPING = {
    "크게": 1.0,
    "작게": 0.7
}

REGION_SCALE = {
    "left_eye": 1.7,
    "right_eye": 1.7,
    "mouth": 1.7,
    "nose": 1.8,
    "chin": 1.6,
    "left_cheek": 2.0,
    "right_cheek": 2.0
}

def warp_region_tps(image, src_points, scale=1.5, blur_size=15, blur_sigma=5):
    src_points = np.array(src_points, dtype=np.float32)
    center = np.mean(src_points, axis=0)
    translated = src_points - center
    scaled = translated * scale
    dst_points = scaled + center
    dst_points = dst_points.astype(np.float32)

    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
    tps.estimateTransformation(np.array([dst_points]), np.array([src_points]), matches)
    warped_image = tps.warpImage(image)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_points), 255)
    if blur_size > 0:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_sigma)
    mask_3ch = cv2.merge([mask]*3).astype(np.float32) / 255.0

    warped_image = warped_image.astype(np.float32)
    image = image.astype(np.float32)
    blended = warped_image * mask_3ch + image * (1 - mask_3ch)
    return np.clip(blended, 0, 255).astype(np.uint8)

def warp_region_with_soft_blend(image, landmarks, indexes, scale=1.2):
    points = np.array([landmarks[i] for i in indexes])
    x, y, w, h = cv2.boundingRect(points)
    cx, cy = x + w // 2, y + h // 2
    new_w, new_h = int(w * scale), int(h * scale)

    roi = image[y:y+h, x:x+w].copy()
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_x = cx - new_w // 2
    new_y = cy - new_h // 2

    h_img, w_img = image.shape[:2]
    x1, y1 = max(new_x, 0), max(new_y, 0)
    x2 = min(new_x + new_w, w_img)
    y2 = min(new_y + new_h, h_img)
    roi_blended = resized[:y2 - y1, :x2 - x1]

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    pad = 20
    mask[pad:-pad, pad:-pad] = 1.0
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    mask_3ch = cv2.merge([mask]*3)

    fg = roi_blended.astype(np.float32)
    bg = image[y1:y2, x1:x2].astype(np.float32)
    blended = fg * mask_3ch + bg * (1 - mask_3ch)
    image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return image

def warp_nose_with_soft_blend(image, landmarks, scale=1.6):
    return warp_region_with_soft_blend(image, landmarks, FACIAL_REGIONS["nose"], scale)

def apply_modification(image, landmarks, modifications):
    output = image.copy()
    for region_key, action in modifications.items():
        indexes = FACIAL_REGIONS.get(region_key)
        if region_key == "볼":
            for cheek in ["left_cheek", "right_cheek"]:
                indexes = FACIAL_REGIONS[cheek]
                base_scale = REGION_SCALE.get(cheek, 1.0)
                action_scale = SCALE_MAPPING.get(action, 1.0)
                scale = base_scale * action_scale
                output = warp_region_with_soft_blend(output, landmarks, indexes, scale)
            continue

        if indexes is None:
            print(f"[경고] '{region_key}'는 알 수 없는 부위입니다.")
            continue

        base_scale = REGION_SCALE.get(region_key, 1.0)
        action_scale = SCALE_MAPPING.get(action, 1.0)
        scale = base_scale * action_scale

        if region_key == "nose":
            output = warp_nose_with_soft_blend(output, landmarks, scale)
        elif region_key in ["chin", "left_cheek", "right_cheek"]:
            output = warp_region_with_soft_blend(output, landmarks, indexes, scale)
        else:
            region_coords = [landmarks[i] for i in indexes]
            output = warp_region_tps(output, region_coords, scale=scale)

    return output

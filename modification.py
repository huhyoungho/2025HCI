import cv2
import numpy as np

FACIAL_REGIONS = {
    "left_eye": [33, 133, 160, 159, 158, 157, 173, 246],
    "right_eye": [362, 263, 387, 386, 385, 384, 398, 466],
    "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
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

REGION_SCALE = {
    "left_eye": 1.5,
    "right_eye": 1.5,
    "mouth": 1.2,
    "nose": 1.5,
    "chin": 1.0,
    "left_cheek": 1.05,
    "right_cheek": 1.05
}

def warp_region_tps(image, src_points, scale=1.5):
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

    # Create mask for destination region and smooth it
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_points), 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 15)

    # Convert to float for blending
    mask_3ch = cv2.merge([mask]*3).astype(np.float32) / 255.0
    warped_image = warped_image.astype(np.float32)
    image = image.astype(np.float32)

    blended = warped_image * mask_3ch + image * (1 - mask_3ch)
    return np.clip(blended, 0, 255).astype(np.uint8)

def apply_modification(image, landmarks, modifications):
    output = image.copy()
    for region_key, action in modifications.items():
        if region_key == "볼":
            for cheek in ["left_cheek", "right_cheek"]:
                indexes = FACIAL_REGIONS[cheek]
                base_scale = REGION_SCALE.get(cheek, 1.0)
                action_scale = SCALE_MAPPING.get(action, 1.0)
                scale = base_scale * action_scale
                region_coords = [landmarks[i] for i in indexes]
                output = warp_region_tps(output, region_coords, scale=scale)
            continue

        if region_key not in FACIAL_REGIONS:
            continue

        indexes = FACIAL_REGIONS[region_key]
        base_scale = REGION_SCALE.get(region_key, 1.0)
        action_scale = SCALE_MAPPING.get(action, 1.0)
        scale = base_scale * action_scale
        region_coords = [landmarks[i] for i in indexes]
        output = warp_region_tps(output, region_coords, scale=scale)
    return output

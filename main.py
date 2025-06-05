# main.py
from capture import capture_face_with_guidelines
from user_input import get_user_modifications
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification
from edge_detection import apply_edge_detection
from caricature_output import generate_binary_caricature
from preprocessing import preprocess_image
from edge_postprocessing import edge_postprocessing
from cartoon import cartoon_effect
import cv2
import numpy as np

if __name__ == "__main__":
    img = capture_face_with_guidelines()
    if img is None:
        print("⚠️ 캡처 실패 또는 취소되었습니다.")
        exit()

    preprocessed_img = preprocess_image(img)
    modifications = get_user_modifications()

    try:
        landmarks = extract_landmarks_mediapipe(preprocessed_img)
    except ValueError as e:
        print("❌ 얼굴 인식 실패:", e)
        exit()

    modified_img = apply_modification(preprocessed_img, landmarks, modifications)
    cartoon_img = cartoon_effect(modified_img)
    canny = apply_edge_detection(cartoon_img)
    #post_canny=edge_postprocessing(canny)
    final = generate_binary_caricature(canny)

    # 박스 외부 마스킹
    height, width = final.shape[:2]
    box_size = 200
    center_x, center_y = width // 2, height // 2
    x1, y1 = center_x - box_size // 2, center_y - box_size // 2
    x2, y2 = center_x + box_size // 2, center_y + box_size // 2

    mask = np.zeros_like(final, dtype=np.uint8)
    mask[y1:y2, x1:x2] = final[y1:y2, x1:x2]
    final = mask

    # 결과 출력 및 저장
    cv2.imshow("원본 이미지", img)
    cv2.imshow("Cartoon Image", cartoon_img)
    cv2.imshow("Edge Detection", canny)
    cv2.imshow("Caricature Emoji", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

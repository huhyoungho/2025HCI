from capture import capture_face_with_guidelines
from user_input import get_user_modifications
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification
from cartoon import cartoon_effect
from processing import processing_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = capture_face_with_guidelines()
    if img is None:
        print("⚠️ 캡처 실패 또는 취소되었습니다.")
        exit()

    modifications = get_user_modifications()   #사용자 입력 받음

    try:
        landmarks = extract_landmarks_mediapipe(img)
    except ValueError as e:
        print("❌ 얼굴 인식 실패:", e)
        exit()

    processed_img=processing_image(img,0.5)
    modified_img = apply_modification(processed_img, landmarks, modifications)
    # 만화 효과 적용
    cartooned = cartoon_effect(modified_img)

    # 결과 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(cartooned, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

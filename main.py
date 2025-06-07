from capture import capture_face_with_guidelines
from user_input import get_user_modifications
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification, rotate_region, FACIAL_REGIONS
from cartoon import cartoon_effect
from preprocessing import preprocessing_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img = capture_face_with_guidelines()
    if img is None:
        print("⚠️ 캡처 실패 또는 취소되었습니다.")
        return

    modifications, rotate_eyes = get_user_modifications()   # 사용자 입력 받음
    landmarks = extract_landmarks_mediapipe(img)
    processed_img = preprocessing_image(img, 0.8)
    modified_img = apply_modification(processed_img, landmarks, modifications)

    # 눈 회전 적용 (사용자가 y/Y 선택 시에만)
    if rotate_eyes:
        modified_img = rotate_region(modified_img, landmarks, FACIAL_REGIONS["left_eye"], 90)
        modified_img = rotate_region(modified_img, landmarks, FACIAL_REGIONS["right_eye"], -90)

    cartooned = cartoon_effect(modified_img)

    # 원본, 결과 이미지 저장 경로 지정
    origin_dir = 'assets/origin'
    result_dir = 'assets/result'
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    origin_path = os.path.join(origin_dir, 'origin.png')
    result_path = os.path.join(result_dir, 'result.png')

    # 원본 저장
    cv2.imwrite(origin_path, img)
    # 결과 저장
    cv2.imwrite(result_path, cartooned)

    # 결과 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(cartooned, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    print(f'원본: {origin_path}\n결과: {result_path} 에 저장됨')

if __name__ == "__main__":
    main()
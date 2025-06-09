from capture import capture_face_with_guidelines
from user_input import get_user_modifications
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification, rotate_region, FACIAL_REGIONS
from cartoon import cartoon_effect
from preprocessing_eq import preprocessing_image_eq
from preprocessing_st import preprocessing_image_st
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
    processed_img_eq = preprocessing_image_eq(img, 0.8)
    processed_img_st=preprocessing_image_st(img,0.8)
    modified_img_eq = apply_modification(processed_img_eq, landmarks, modifications)
    modified_img_st = apply_modification(processed_img_st, landmarks, modifications)
    # 눈 회전 적용 (사용자가 y/Y 선택 시에만)
    if rotate_eyes:
        modified_img_eq = rotate_region(modified_img_eq, landmarks, FACIAL_REGIONS["left_eye"], 90)
        modified_img_eq= rotate_region(modified_img_eq, landmarks, FACIAL_REGIONS["right_eye"], -90)

        modified_img_st= rotate_region(modified_img_st, landmarks, FACIAL_REGIONS["left_eye"], 90)
        modified_img_st= rotate_region(modified_img_st, landmarks, FACIAL_REGIONS["right_eye"], -90)

    cartooned_eq= cartoon_effect(modified_img_eq)
    cartooned_st= cartoon_effect(modified_img_st)
    # 원본, 결과 이미지 저장 경로 지정
    origin_dir = 'assets/origin'
    result_dir_eq = 'assets/result'
    result_dir_st='assets/result'
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(result_dir_eq, exist_ok=True)
    os.makedirs(result_dir_st, exist_ok=True)
    origin_path = os.path.join(origin_dir, 'origin.png')
    result_path_eq = os.path.join(result_dir_eq, 'result_eq.png')
    result_path_st= os.path.join(result_dir_st, 'result_st.png')

    # 원본 저장
    cv2.imwrite(origin_path, img)
    # 결과 저장
    cv2.imwrite(result_path_eq, cartooned_eq)
    cv2.imwrite(result_path_st, cartooned_st)

    # 결과 출력
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cartooned_eq, cv2.COLOR_BGR2RGB))
    plt.title("Histogram_Equaliztion")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cartooned_st, cv2.COLOR_BGR2RGB))
    plt.title("Histogram_Stretching")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
# test.py
import cv2
import numpy as np
from fer import FER
from sklearn.metrics.pairwise import cosine_similarity
import os

ORIGINAL_IMAGE_PATH = 'assets/origin/origin.png'
CARICATURE_IMAGE_PATH_EQ = 'assets/result/result_eq.png'
CARICATURE_IMAGE_PATH_ST = 'assets/result/result_st.png'

def get_emotion_vector(image):
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(image)
    if not result:
        return None
    return np.array(list(result[0]['emotions'].values()))

def emotion_consistency_score(img1, img2):
    vec1 = get_emotion_vector(img1)
    vec2 = get_emotion_vector(img2)
    if vec1 is None or vec2 is None:
        print('감정 인식 실패')
        return None
    return cosine_similarity([vec1], [vec2])[0][0]

def main():
    if not (os.path.exists(ORIGINAL_IMAGE_PATH) and os.path.exists(CARICATURE_IMAGE_PATH_EQ) and os.path.exists(CARICATURE_IMAGE_PATH_ST)):
        print('평가용 이미지가 존재하지 않습니다. main.py를 먼저 실행하세요.')
        return None, None

    img1 = cv2.imread(ORIGINAL_IMAGE_PATH)
    img_eq = cv2.imread(CARICATURE_IMAGE_PATH_EQ)
    img_st = cv2.imread(CARICATURE_IMAGE_PATH_ST)

    if img1 is None or img_eq is None or img_st is None:
        print('이미지 로드 실패')
        return None, None

    score_eq = emotion_consistency_score(img1, img_eq)
    score_st = emotion_consistency_score(img1, img_st)
    print(f'Equalization Score: {score_eq:.4f}')
    print(f'Stretching Score: {score_st:.4f}')

    return score_eq, score_st

if __name__ == '__main__':
    main()

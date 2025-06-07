import cv2
import numpy as np
from fer import FER
from sklearn.metrics.pairwise import cosine_similarity
import os

# 저장된 원본/캐리커처 이미지 경로
ORIGINAL_IMAGE_PATH = 'assets/origin/origin.png'
CARICATURE_IMAGE_PATH = 'assets/result/result.png'

def get_emotion_vector(image):
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(image)
    if not result:
        return None
    # FER 라이브러리의 감정 확률 벡터 추출
    return np.array(list(result[0]['emotions'].values()))

def emotion_consistency_score(img1, img2):
    vec1 = get_emotion_vector(img1)
    vec2 = get_emotion_vector(img2)
    if vec1 is None or vec2 is None:
        print('감정 인식 실패')
        return None
    # 코사인 유사도
    return cosine_similarity([vec1], [vec2])[0][0]

def main():
    if not (os.path.exists(ORIGINAL_IMAGE_PATH) and os.path.exists(CARICATURE_IMAGE_PATH)):
        print('평가용 이미지가 존재하지 않습니다. main.py를 먼저 실행하세요.')
        return
    img1 = cv2.imread(ORIGINAL_IMAGE_PATH)
    img2 = cv2.imread(CARICATURE_IMAGE_PATH)
    if img1 is None or img2 is None:
        print('이미지 로드 실패')
        return
    score = emotion_consistency_score(img1, img2)
    print(f'Emotion Consistency Score: {score}')

if __name__ == '__main__':
    main()

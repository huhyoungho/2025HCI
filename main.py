from utils.face_capture import capture_face_with_guide
from utils.preprocessing import preprocess_image
from utils.landmark_detection import detect_landmarks_mediapipe
from utils.exaggeration import scale_feature
from utils.edge_detection import canny_edge
from utils.emoji_generator import generate_emoji

def main():
    face_img = capture_face_with_guide()
    preprocessed = preprocess_image(face_img)

    landmarks = detect_landmarks_mediapipe(face_img)
    if not landmarks:
        print("얼굴 랜드마크 탐지 실패")
        return

    # 예시: 눈 영역 과장 (임의의 좌표 10개만 사용)
    eye_points = landmarks[33:43]  # 오른쪽 눈 좌표 예시
    exaggerated_img = scale_feature(face_img, eye_points, 1.3)

    edge = canny_edge(preprocessed)
    emoji = generate_emoji(exaggerated_img, cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR))

    cv2.imshow("Final Emoji", emoji)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

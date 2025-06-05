import cv2
import mediapipe as mp

def capture_face_with_guidelines():
    cap = cv2.VideoCapture(0)
    captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 안내선, 텍스트, 박스 모두 제거
        # 얼굴 검출만 수행
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        face_detected = False
        if results.detections:
            face_detected = True

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)
        # 아무 키나 누르면(얼굴만 있으면) 캡처
        if key != -1 and face_detected:
            captured = frame.copy()
            break
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

import cv2
import numpy as np

def capture_face_with_guidelines():
    cap = cv2.VideoCapture(0)
    width, height = 640, 480

    box_size = 200
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 안내선 그리기
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, "Align your face inside the box", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):  # 'c'를 눌러 캡처
            # 캡처된 프레임 저장
            captured = frame.copy()

            # 마스크 생성: 박스 영역은 1, 나머지는 0
            mask = np.zeros_like(captured, dtype=np.uint8)
            x1, y1 = top_left
            x2, y2 = bottom_right
            mask[y1:y2, x1:x2] = captured[y1:y2, x1:x2]

            # 박스 밖은 검정색 처리된 이미지 반환
            captured = mask
            break

        elif key == 27:  # ESC
            captured = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

import cv2

def capture_face_with_guide():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 안내선 그리기
        h, w = frame.shape[:2]
        box_size = 300
        top_left = (w // 2 - box_size // 2, h // 2 - box_size // 2)
        bottom_right = (w // 2 + box_size // 2, h // 2 + box_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):  # c 키를 눌러 캡처
            face_img = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_img

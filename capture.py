import cv2
import mediapipe as mp

def capture_face_with_guidelines():
    cap = cv2.VideoCapture(0)
    width, height = 640, 480

    box_size = 200
    center_x, center_y = width // 2, height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)

    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 안내선 박스
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, "Align your face and press 'c'", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        face_in_box = False
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width + bbox.width * width / 2)
                y = int(bbox.ymin * height + bbox.height * height / 2)
                if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                    face_in_box = True
                    cv2.putText(frame, "Face OK", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)
        if key == ord('c') and face_in_box:
            captured = frame.copy()
            break
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

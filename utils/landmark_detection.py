import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks_mediapipe(img):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return []
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        return [(int(p.x * w), int(p.y * h)) for p in landmarks]
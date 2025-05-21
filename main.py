from capture import capture_face_with_guidelines
from user_input import get_user_modifications
from landmark_extraction import extract_landmarks_mediapipe
from modification import apply_modification
from edge_detection import apply_edge_detection
from caricature_output import generate_binary_caricature
import cv2

if __name__ == "__main__":
    img = capture_face_with_guidelines()
    modifications = get_user_modifications()
    landmarks = extract_landmarks_mediapipe(img)
    modified_img = apply_modification(img, landmarks, modifications)
    canny= apply_edge_detection(modified_img)
    final = generate_binary_caricature(canny)

    cv2.imshow("Caricature Emoji", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
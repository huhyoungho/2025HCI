import cv2

def generate_emoji(final_img, edge_img):
    blend = cv2.addWeighted(final_img, 0.7, edge_img, 0.3, 0)
    return blend
import numpy as np
def generate_binary_caricature(edge_img):
    binary = np.where(edge_img > 0, 0, 255).astype(np.uint8)
    return binary

# caricature = generate_binary_caricature(canny)
# cv2.imshow("Final Caricature", caricature)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

def non_max_suppression(edges, gradient_x, gradient_y):
    """
    Non-Max Suppression 구현 (간단 버전)
    edges: 이진 엣지 맵 (0 or 255)
    gradient_x, gradient_y: sobel gradient x, y
    
    return: NMS 처리된 엣지 이미지
    """
    M, N = edges.shape
    Z = np.zeros((M,N), dtype=np.uint8)

    angle = np.arctan2(gradient_y, gradient_x) * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # 각도별로 주변 픽셀과 비교
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = edges[i, j+1]
                    r = edges[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = edges[i+1, j-1]
                    r = edges[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = edges[i+1, j]
                    r = edges[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = edges[i-1, j-1]
                    r = edges[i+1, j+1]

                if (edges[i,j] >= q) and (edges[i,j] >= r):
                    Z[i,j] = edges[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z


def edge_postprocessing(edge_img):
    # 1. Sobel gradient 계산 (Non-Max Suppression용)
    sobelx = cv2.Sobel(edge_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edge_img, cv2.CV_64F, 0, 1, ksize=3)

    # 2. Non-Max Suppression으로 엣지 선명하게 유지
    nms = non_max_suppression(edge_img, sobelx, sobely)

    # 3. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # 노이즈 제거를 위해 Erode (침식) 1회
    eroded = cv2.erode(nms, kernel, iterations=1)

    # 선 굵게 하고 끊어진 선 잇기 위해 Dilate (팽창) 2회
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    return dilated

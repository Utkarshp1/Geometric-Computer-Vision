import cv2

import numpy as np

def draw_lines_on_image(img , points):
    points = points.astype(np.int32)

    for i in range(0, len(points), 2):
        img = cv2.line(img, points[i], points[i+1], (0, 255, 0), 2)

    return img
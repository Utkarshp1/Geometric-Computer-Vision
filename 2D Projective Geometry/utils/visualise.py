import cv2

# def draw_lines_on_image(img: np.ndarray, points : np.ndarray):
#     points = points.astype(np.int32)
#     img = cv2.line(img, points[0], points[1], (0, 255, 0), 2)
#     img = cv2.line(img, points[2], points[3], (0, 255, 0), 2)
#     img = cv2.line(img, points[4], points[5], (0, 255, 0), 2)
#     img = cv2.line(img, points[6], points[7], (0, 255, 0), 2)
#     img = cv2.line(img, points[8], points[9], (0, 255, 0), 2)
#     cv2.imwrite('img_draw.png', img)
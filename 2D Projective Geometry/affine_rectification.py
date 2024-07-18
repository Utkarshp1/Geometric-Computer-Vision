'''
    To remove perspective distortion from an image or equivalently
    affinely rectify the image we need to do the following:
        1. Identify the vanishing line in the image (or the imaged line
            at infinity.) Let this be represented by (l1, l2, l3)^T.
        2. The homography matrix which maps this line to the canonical
            position (i.e., (0, 0, 1)^T) of line at infinity in a
            Euclidean plain is given as:
                [[1, 0, 0],
                 [0, 1, 0],
                 [l1, l2, l3]]
            assuming l3 != 0.

    To identify the vanishing lines, we do the following:
        1. Identify the two pairs of parallel line in the real-world
            and the corresponding lines in the image.
        2. Identify the intersection points of these pairs of lines in
            the image. The points should lie on line at infinity because
            they are the intersection points of parallel lines.
        3. Find the line passing through these two intersection points.
            This should be the vanishing line.
'''

import os

import cv2
import numpy as np

from utils import *

def affine_rectify(img, points, dst_shape):
    lines = construct_lines_from_points(points, method="cross-product")
    intersect_points = find_intersection_of_lines(lines, method="cross-product")
    l_infinity_image = construct_lines_from_points(intersect_points, method="cross-product")
    l_infinity_image = convert_homo_to_inhomo_coor(l_infinity_image)

    affine_rect_matrix = np.eye(3)
    affine_rect_matrix[2] = l_infinity_image

    return cv2.warpPerspective(img, affine_rect_matrix, dst_shape)

if __name__ == '__main__':
    annotations = load_annotations('../16-822/assignment1/data/annotation/q1_annotation.npy')
    img_dir = '../16-822/assignment1/data/q1'
    dst_dir = 'afine_rectified_images'
    os.makedirs(dst_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_name))
        points = extract_points_from_annotations(annotations,
                    image_name=img_name.split('.')[0])
        train_points = filter_lines(points, mode='train')
        warped_img = affine_rectify(
            img, train_points,
            (img.shape[1], img.shape[0])
        )
        cv2.imwrite(os.path.join(dst_dir, img_name), warped_img)
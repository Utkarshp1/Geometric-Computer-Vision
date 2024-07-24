'''
    Two-step procedure for metric rectification:
        1. First do an affine rectification of the image using two pairs
            of parallel lines in the image.
        2. For metric rectification identify two pairs of orthogonal
            lines in the affine rectified image. If l' = [l1', l2', l3']
            and m' = [m1', m2', m3'] represent one such pair, then using
            the orthogonality constraint we obtain:
                (l1'*m1', l1'*m2' + l2'*m1', l2'*m2')s = 0
            where s = (s11, s12, s22)^T represents the elements of the
            symmetric matrix S. This S = KK^T where K is the 2x2 top-left
            sub-matrix of an affine transformation matrix. Now, any K can
            be written as K = UDV^T = (UV)(VDV^T). Now ignoring the pure
            rotation part (UV) because metric rectification can be done
            uptill a similarity transformation. Substituting, K = VDV^T
            in S, we get S = VD^2V^T. Using SVD on S, we can get D^2 and
            V. Using this we construct K = VDV^T. Therefore, the final
            metric refinement matrix will be 3x3 identity matrix with the
            top-left 2x2 sub-matrix replaced by K.
'''

import os

import cv2
import scipy
import numpy as np

import utils
import affine_rectification as affine_rect
import utils.geometric
import utils.test

def construct_metric_rect_matrix_from_2_ortho_lines(ortho_lines):
    matrix = []
    for i in range(0, len(ortho_lines), 2):
        l1_, l2_ = ortho_lines[i, 0], ortho_lines[i, 1]
        m1_, m2_ = ortho_lines[i+1, 0], ortho_lines[i+1, 1]

        matrix.append([l1_*m1_, l1_*m2_ + l2_*m1_, l2_*m2_])

    return np.stack(matrix)

def metric_rectify(img, ortho_line_pts, parallel_line_pts):
    ortho_lines = utils.geometric.construct_lines_from_points(
        ortho_line_pts, method='cross-product')

    affine_rect_img, A = affine_rect.affine_rectify(img.copy(), parallel_line_pts)
    affine_rect_ortho_lines = utils.geometric.warp_line(ortho_lines, A)

    metric_rect_matrix = construct_metric_rect_matrix_from_2_ortho_lines(
        affine_rect_ortho_lines)
    s = scipy.linalg.null_space(metric_rect_matrix)
    s = utils.geometric.convert_homo_to_inhomo_coor(s.T).T
    S = np.array([[s[0, 0], s[1, 0]], [s[1, 0], s[2, 0]]])
    U, sigma, V_T = np.linalg.svd(S)

    sigma = np.sqrt(sigma)
    temp = np.eye(2)
    temp[0, 0] = sigma[0]
    temp[1, 1] = sigma[1]
    K = np.linalg.inv(np.dot(U, temp).dot(V_T))
    metric_rect_matrix = np.eye(3)
    metric_rect_matrix[:2, :2] = K

    metric_rect_matrix, dst_shape = utils.geometric.fit_warped_image_in_frame(
        affine_rect_img.shape, metric_rect_matrix)
    interp_flag = utils.geometric.determine_interpolation(img.shape, dst_shape)

    # C_star_inf = np.zeros((3, 3))
    # C_star_inf[:2, :2] = S

    # U, sigma, V_T = np.linalg.svd(C_star_inf)
    # # Make sigma identity
    # s1 = np.sqrt(sigma[0])
    # s2 = np.sqrt(sigma[1])
    # temp = np.eye(3)
    # temp[0, 0] = s1
    # temp[1, 1] = s2
    # metric_rect_matrix = np.linalg.inv(np.dot(U, temp).dot(V_T))

    warped_img = cv2.warpPerspective(
        affine_rect_img, metric_rect_matrix, dst_shape[::-1], flags=interp_flag)

    return warped_img, np.dot(metric_rect_matrix, A)

if __name__ == '__main__':
    parallel_line_annotations = utils.loader.load_annotations(
        '../16-822/assignment1/data/annotation/q1_annotation.npy')
    ortho_line_annotations = utils.loader.load_annotations(
        '../16-822/assignment1/data/annotation/q2_annotation.npy')
    img_dir = '../16-822/assignment1/data/q1'
    dst_dir = 'metric_rectified_images'
    os.makedirs(dst_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        print(img_name)
        img = cv2.imread(os.path.join(img_dir, img_name))
        parallel_line_points = utils.loader.extract_points_from_annotations(
            parallel_line_annotations, image_name=img_name.split('.')[0])
        ortho_line_points = utils.loader.extract_points_from_annotations(
            ortho_line_annotations, image_name=img_name.split('.')[0])
        train_parallel_points = utils.loader.filter_lines(
            parallel_line_points, mode='train')
        train_ortho_points = utils.loader.filter_lines(
            ortho_line_points, mode='train')

        warped_img, M = metric_rectify(img, train_ortho_points, train_parallel_points)
        cv2.imwrite(os.path.join(dst_dir, img_name), warped_img)

        test_points = utils.loader.filter_lines(
            ortho_line_points, mode='test')
        utils.test.test_angles_between_lines(test_points, M)
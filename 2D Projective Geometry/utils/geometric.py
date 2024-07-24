import operator
from functools import reduce

import cv2
import numpy as np

def convert_inhomo_to_homo_coor(arr):
    '''
        Will add 1 as the last column.
    '''
    return np.hstack([arr, np.expand_dims(np.ones(len(arr)), axis=-1)])

def convert_homo_to_inhomo_coor(arr, retain_last_col=True):
    '''
        Will divide all the columns with the last column.
    '''
    inhomo_coor = arr/arr[:, -1:]

    return inhomo_coor if retain_last_col else inhomo_coor[:, :2]


def construct_lines_from_points(points : np.ndarray, method="cross-product"):
    '''
        Naive Method:
            Given two (x1, y1) and (x2, y2), the equation of line through
            these points is given by:
                (y - y1)/(x - x1) = (y1 - y2)/(x1 - x2)
            Equivalently,
                (y1 - y2)x + (x2 - x1)y + (x1 - x2)y1 - (y1 - y2)x1 = 0

        Cross Product Method:
            Given two p1 and p2, the line passing through both of them
            is given by p1 x p2, where are p1 and p2 are points in 2D
            homogeneous coordinates (i.e., are 3D vectors).

        Linear System of Equations Method:
            Assume that the equation of line is given by x + ay + b = 0
            where a and b are unknown. Substitute x and y for each point
            to get 2 linear equations which can be solved using Gaussian
            Elimination.
                np.dot([[y1 , 1],[y2,  1]], [a, b]) = [-x1, -x2]

        Arguments:
        ---------
            points: Each consecutive pair of points represent a line.
    '''

    if method == "naive":
        lines = []
        for i in range(0, len(points), 2):
            x1, y1, x2, y2 = points[i, 0], points[i, 1], points[i+1, 0], points[i+1, 1]
            lines.append([
                (y1 - y2),
                (x2 - x1),
                (x1 - x2)*y1 - (y1 - y2)*x1
            ])

        return np.array(lines)

    elif method == "cross-product":
        if points.shape[1] == 2:
            points = convert_inhomo_to_homo_coor(points)
        points1 = points[range(0, len(points), 2)]
        points2 = points[range(1, len(points), 2)]

        return np.cross(points1, points2)

    elif method == "linear-system":
        lines = []
        for i in range(0, len(points), 2):
            x1, y1, x2, y2 = points[i, 0], points[i, 1], points[i+1, 0], points[i+1, 1]
            a, b = np.linalg.solve(np.array([[y1, 1], [y2, 1]]), [-x1, -x2])
            lines.append([1, a, b])

        return np.array(lines)

def find_intersection_of_lines(lines : np.ndarray, method="cross-product"):
    '''
        Cross Product Method:
            Given two lines l1 and l2, the intersection point of the two
            lines is given by l1 x l2, where l1 and l2 are the homogeneous
            vectors (3D) of the lines.

        Linear System of Equations method:
            Finding intersection of two lines is similar to solving Ax=b.

        Arguments:
        ---------
            lines: Find intersection of consecutive lines in `lines`
    '''

    if method == "linear-system":
        intersection_points = []
        for i in range(0, len(lines), 2):
            intersection_points.append(np.linalg.solve(lines[i:i+2, :2], -lines[i:i+2, 2]))

        return np.stack(intersection_points)

    elif method == "cross-product":
        lines1 = lines[range(0, len(lines), 2)]
        lines2 = lines[range(1, len(lines), 2)]

        return np.cross(lines1, lines2)

def warp_line(lines, H):
    '''
        Under the point transformation x' = Hx, a line transforms as
            l' = H^(-T)l

        Arguments:
        ---------
            - lines : A NumPy array of shape (N, 3) where N is the number
                of lines. The lines are represented in homogeneous
                coordinates.
            - H : 2D Transformation matrix of shape (3, 3).

        Returns:
        -------
            NumPy array of shape (N, 3) where N is the number of lines.
    '''

    return np.dot(np.linalg.inv(H).T, lines.T).T

def warp_points(points, H, return_form="homo"):
    '''
        Arguments:
        ---------
            - points : A NumPy array of shape (N, 2) where N is the
                number of points
            - H : 2D Transformation matrix of shape (3, 3).
            - return_form : If set to `homo` returns the warped points
                in homogeneous form. Otherwise if set to `inhomo`
                returns warped points in inhomogeneous form.
    '''

    homo_points = convert_inhomo_to_homo_coor(points)
    warped_points = np.dot(H, homo_points.T).T
    if return_form == "inhomo":
        return convert_homo_to_inhomo_coor(warped_points, retain_last_col=False)
    elif return_form == "homo":
        return warped_points

def fit_warped_image_in_frame(img_shape, H):
    '''
        This function will modify the transformation matrix to translate
        the image such that the contents of the original image can be
        brought into the frame. Also, it will decide the shape of the
        warped image such that all the contents of the original image
        are visible in the warped image.

        Arguments:
        ---------
            - img_shape : Shape of the image to be warped in terms of
                (H, W).
            - H : 2D Transformation matrix of shape (3, 3).

        Returns:
        -------
            - H : New translation-fixed 2D transformation matrix.
            - dst_shape : New shape of the warped image.
    '''

    corner_points = np.array([
        [0, 0],
        [img_shape[1], 0],
        [0, img_shape[0]],
        [img_shape[1], img_shape[0]],
    ])

    warped_points = warp_points(corner_points, H, return_form="inhomo")

    x_min, x_max = warped_points[:, 0].min(), warped_points[:, 0].max()
    y_min, y_max = warped_points[:, 1].min(), warped_points[:, 1].max()

    # Bring (x_min, y_min) to (0, 0) so that the image fits in the frame
    tx, ty = -x_min, -y_min
    H[0, 2], H[1, 2] = tx, ty

    dst_shape = (int(y_max - y_min), int(x_max - x_min))

    return H, dst_shape

def determine_interpolation(src_img_shape, dst_img_shape):
    '''
        According to cv2 guidelines, if the image is being shrunken
        then use cv2.INTER_AREA interpolation. Otherwise, if the image
        is enlarged then use cv2.INTER_LINEAR interpolation.
    '''

    return (cv2.INTER_AREA if
            reduce(operator.mul, src_img_shape) > reduce(operator.mul, dst_img_shape) else
            cv2.INTER_LINEAR)

def get_cosine_of_angle_between_lines(lines):
    '''
        Computes the cosine of the angle between consecutive lines in
        the array.

        Arguments:
        ---------
            - lines: A NumPy array of shape (N, 3) i.e., the lines are
                in homogenoeus form.
    '''

    angles = []

    for i in range(0, len(lines), 2):
        dot_prod = np.dot(lines[i, :2], lines[i+1, :2])
        l_norm = np.linalg.norm(lines[i, :2], ord=2)
        m_norm = np.linalg.norm(lines[i+1, :2], ord=2)
        angles.append(dot_prod / (l_norm * m_norm))

    return angles

if __name__ == '__main__':
    points = np.array(
        [[-1, 0],
         [1, 1],
         [0, 1],
         [1, 0]]
    )

    print(construct_lines_from_points(points, method="naive"))
    print(construct_lines_from_points(points, method="cross-product"))
    print(construct_lines_from_points(points, method="linear-system"))
    print(find_intersection_of_lines(construct_lines_from_points(points), method="cross-product"))
    print(find_intersection_of_lines(construct_lines_from_points(points), method="linear-system"))
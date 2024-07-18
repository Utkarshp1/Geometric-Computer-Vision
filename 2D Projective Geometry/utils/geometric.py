import numpy as np

def convert_inhomo_to_homo_coor(arr):
    '''
        Will add 1 as the last column.
    '''
    return np.hstack([arr, np.expand_dims(np.ones(len(arr)), axis=-1)])

def convert_homo_to_inhomo_coor(arr):
    '''
        Will divide all the columns with the last column.
    '''
    return arr/arr[:, -1:]


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
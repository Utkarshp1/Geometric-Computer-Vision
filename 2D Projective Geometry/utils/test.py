from . import geometric

def test_angles_between_lines(points, H):
    '''
    '''
    lines = geometric.construct_lines_from_points(points)
    unwarped_angles = geometric.get_cosine_of_angle_between_lines(lines)

    warped_lines = geometric.warp_line(lines, H)
    warped_angles = geometric.get_cosine_of_angle_between_lines(warped_lines)

    for unwarped_angle, warped_angle in zip(unwarped_angles, warped_angles):
        print(f'Cosine of angle Before: {unwarped_angle} After: {warped_angle}')
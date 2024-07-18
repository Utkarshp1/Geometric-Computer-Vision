import numpy as np

def load_annotations(annotation_path):
    with open(annotation_path,'rb') as f:
        annotations = np.load(f, allow_pickle=True)

    return annotations

def extract_points_from_annotations(annotations, image_name):
    return annotations.item().get(image_name)

def filter_lines(points, mode='train'):
    return points[:8] if mode == 'train' else points[8:]
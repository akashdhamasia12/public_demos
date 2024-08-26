import os
import numpy as np


class PreprocessingMeta:
    def __init__(self, recording_dir: str, reference_utm: dict, intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 reprojection_error: float, recording_meta: dict,
                 frame_dist: int, img_dicts: list) -> None:
        self.version = '1.4'
        self.recording_dir = os.path.basename(os.path.abspath(recording_dir))
        self.reference_utm = reference_utm
        self.intrinsic_matrix = intrinsic_matrix.tolist()
        self.distortion_coefficients = dist_coeffs.tolist()
        self.reprojection_error = reprojection_error
        self.image_width = recording_meta['image_width']
        self.image_height = recording_meta['image_height']
        self.frame_rate = recording_meta['frame_rate']
        self.frame_distance = frame_dist
        self.images = [{key: value.tolist() if isinstance(value, np.ndarray) else value for (key, value) in
                        img_dict.items()} for img_dict in img_dicts]

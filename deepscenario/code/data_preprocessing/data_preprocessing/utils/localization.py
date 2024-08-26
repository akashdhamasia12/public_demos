import numpy as np
from scipy.spatial.transform import Rotation
import PIL.Image
import cv2
import open3d as o3d
import logging

from data_preprocessing.utils.generic import denormalize_image_coordinates
from data_preprocessing.utils.features import FeatureConfig, extract_features
from data_preprocessing.raycast_features import raycast_features
from data_preprocessing.utils.matching import MatchingConfig, build_flann_index, match_descriptor_robust
from data_preprocessing.utils.kalman import KalmanFilter


def solve_pnp(pts_3d: np.ndarray, pts_2d: np.ndarray, intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray) -> tuple:
    success, rotation, translation, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, intrinsic_matrix,
                                                                 distCoeffs=dist_coeffs)
    assert success, 'No PnP solution found'
    return np.squeeze(translation), np.squeeze(rotation)


class Localizer:
    def __init__(self, pts_ref_3d: np.ndarray, pts_ref_2d: np.ndarray, desc_ref: np.ndarray,
                 intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray, feature_config: FeatureConfig,
                 matching_config=MatchingConfig) -> None:
        assert len(pts_ref_3d) == len(pts_ref_2d) == len(desc_ref)
        self.pts_ref_3d = pts_ref_3d
        self.pts_ref_2d = pts_ref_2d
        self.desc_ref = desc_ref
        self.flann_idx_ref = build_flann_index(desc_ref, matching_config)
        self.intrinsic_matrix = intrinsic_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_config = feature_config
        self.matching_config = matching_config

    @classmethod
    def initialize_from_frame(cls, frame: PIL.Image.Image, intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray,
                              extrinsic_matrix: np.ndarray, mesh: o3d.geometry.TriangleMesh,
                              feature_config: FeatureConfig) -> 'Localizer':
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        img_height, img_width = img.shape[:2]
        pts_2d, desc = extract_features(img, feature_config)
        pts_2d_denorm = denormalize_image_coordinates(pts_2d, img_width, img_height)
        pts_3d, idxs_hit = raycast_features(pts_2d_denorm, dist_coeffs, intrinsic_matrix, extrinsic_matrix, mesh)
        logging.debug(
            f'Matches with valid projection: {len(idxs_hit)} / {len(pts_2d)} = {len(idxs_hit) / len(pts_2d):.2f}')
        pts_3d = pts_3d[idxs_hit]
        pts_2d = pts_2d[idxs_hit]
        desc = desc[idxs_hit]
        matching_config = MatchingConfig()
        return cls(pts_3d, pts_2d, desc, intrinsic_matrix, dist_coeffs, feature_config, matching_config)

    def localize(self, img_curr: np.ndarray) -> np.ndarray:
        # extract features (note that given image must be distorted)
        pts_curr_2d, desc_curr = extract_features(img_curr, self.feature_config)
        flann_idx_curr = build_flann_index(desc_curr, self.matching_config)

        # match features
        matches = match_descriptor_robust(self.pts_ref_2d, self.desc_ref, self.flann_idx_ref, pts_curr_2d, desc_curr,
                                          flann_idx_curr, self.matching_config)
        pts_curr_3d = self.pts_ref_3d[matches[:, 0]]
        pts_curr_2d = pts_curr_2d[matches[:, 1]]
        pts_curr_2d = denormalize_image_coordinates(pts_curr_2d, img_curr.shape[1], img_curr.shape[0])

        # localize camera
        translation, rotation = solve_pnp(pts_curr_3d, pts_curr_2d, self.intrinsic_matrix, self.dist_coeffs)

        return translation, rotation


def compose_measurement(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    yaw, pitch, roll = Rotation.from_rotvec(rotation).as_euler('ZYX')
    return np.concatenate([translation, [roll, pitch, yaw]])


def decompose_state(state: np.ndarray) -> tuple:
    translation = state[:3]
    roll, pitch, yaw = state[3:6]
    rotation = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_rotvec()
    return translation, rotation


def filter_extrinsics(img_dicts: list) -> list:
    frame_ids = [img_dict['frame_id'] for img_dict in img_dicts]
    if len(frame_ids) > 1:
        assert len(set(np.diff(frame_ids))) == 1, 'Filtering currently requires a constant distance between frames'

    # init filter
    zs = [
        compose_measurement(img_dict['extrinsic_translation'], img_dict['extrinsic_rotation']) for img_dict in img_dicts
    ]
    kf = KalmanFilter(zs[0])

    # forward pass
    x0, P0 = kf.get_state()
    xs, Ps = kf.batch_filter(zs[1:])
    xs = np.concatenate([[x0], xs])
    Ps = np.concatenate([[P0], Ps])

    # backward pass
    xs_smoothed, _ = kf.rts_smoother(xs, Ps)

    # assign smoothed values
    assert len(xs_smoothed) == len(img_dicts)
    for ii, x in enumerate(xs_smoothed):
        translation, rotation = decompose_state(x)
        img_dicts[ii]['extrinsic_translation'] = translation
        img_dicts[ii]['extrinsic_rotation'] = rotation

    return img_dicts


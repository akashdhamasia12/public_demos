import numpy as np
import cv2
from typing import Optional

from data_preprocessing.utils.generic import convert_extrinsic_vectors_to_matrix, denormalize_image_coordinates


def concatenate_matches(matching_pairs: list) -> tuple:
    pts_2d_concat, pts_3d_concat = [], []
    for pair in matching_pairs:
        img_width, img_height = pair.frame2.size
        pts_2d = denormalize_image_coordinates(pair.pts2_2d, img_width, img_height)
        pts_2d_concat.append(pts_2d)
        pts_3d_concat.append(pair.pts_3d)
    pts_2d_concat = np.concatenate(pts_2d_concat)
    pts_3d_concat = np.concatenate(pts_3d_concat)
    return pts_2d_concat, pts_3d_concat


def compute_reprojection_error(pts_2d: np.ndarray, pts_3d: np.ndarray, intrinsic_matrix: np.ndarray,
                               dist_coeffs: np.ndarray, extrinsic_translation: np.ndarray,
                               extrinsic_rotation: np.ndarray) -> float:
    # aligned with https://stackoverflow.com/a/23785171
    assert len(pts_2d) == len(pts_3d), 'Same number of 2d and 3d points required'
    pts_2d_proj, _ = cv2.projectPoints(pts_3d, extrinsic_rotation, extrinsic_translation, intrinsic_matrix, dist_coeffs)
    pts_2d_proj = pts_2d_proj.reshape(-1, 2)
    error_sq = np.sum((pts_2d - pts_2d_proj) ** 2) / len(pts_2d)
    return np.sqrt(error_sq)


def calibrate_extrinsics(pts_2d: np.ndarray, pts_3d: np.ndarray, intrinsic_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> tuple:
    success, extrinsic_rotation, extrinsic_translation, inliers = \
        cv2.solvePnPRansac(pts_3d, pts_2d, intrinsic_matrix, distCoeffs=dist_coeffs)
    assert success, 'No PnP solution found'

    # compute reprojection error
    inliers = np.squeeze(inliers)
    pts_2d_inliers = pts_2d[inliers]
    pts_3d_inliers = pts_3d[inliers]
    extrinsic_translation = np.squeeze(extrinsic_translation)
    extrinsic_rotation = np.squeeze(extrinsic_rotation)
    reprojection_error = compute_reprojection_error(pts_2d_inliers, pts_3d_inliers, intrinsic_matrix, dist_coeffs,
                                                    extrinsic_translation, extrinsic_rotation)

    return extrinsic_translation, extrinsic_rotation, inliers, reprojection_error


def calibrate_full(pts_2d: np.ndarray, pts_3d: np.ndarray, rec_img_size: tuple, fix_principal_point: bool = True,
                   fix_radial_dist: bool = True) -> tuple:
    focal_guess = 3e3
    img_width, img_height = rec_img_size
    intrinsic_matrix_guess = np.array([[focal_guess, 0, img_width / 2 - 0.5],
                                       [0, focal_guess, img_height / 2 - 0.5],
                                       [0, 0, 1]])
    dist_coeffs_guess = np.zeros(5)

    # remove outliers
    extrinsic_translation_guess, extrinsic_rotation_guess, inliers, _ = \
        calibrate_extrinsics(pts_2d, pts_3d, intrinsic_matrix_guess, dist_coeffs_guess)
    inliers = np.squeeze(inliers)
    pts_2d_inliers = np.array([pts_2d[inliers]], dtype=np.float32)
    pts_3d_inliers = np.array([pts_3d[inliers]], dtype=np.float32)

    # compose calibration flags
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K3
    if fix_principal_point:
        flags += cv2.CALIB_FIX_PRINCIPAL_POINT
    if fix_radial_dist:
        flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2

    # optimize intrinsics and extrinsics
    reprojection_error, intrinsic_matrix, dist_coeffs, extrinsic_rotation, extrinsic_translation = \
        cv2.calibrateCamera(pts_3d_inliers, pts_2d_inliers, rec_img_size, cameraMatrix=intrinsic_matrix_guess,
                            distCoeffs=dist_coeffs_guess, rvecs=extrinsic_rotation_guess,
                            tvecs=extrinsic_translation_guess, flags=flags)

    extrinsic_translation = np.squeeze(extrinsic_translation)
    extrinsic_rotation = np.squeeze(extrinsic_rotation)

    return intrinsic_matrix, dist_coeffs, extrinsic_translation, extrinsic_rotation, reprojection_error


def calibrate_camera(matching_pairs: list, intrinsic_matrix: Optional[np.ndarray], dist_coeffs: Optional[np.ndarray],
                     rec_img_size: tuple) -> tuple:
    pts_2d, pts_3d = concatenate_matches(matching_pairs)
    if intrinsic_matrix is not None and dist_coeffs is not None:
        extrinsic_translation, extrinsic_rotation, _, reprojection_error = \
            calibrate_extrinsics(pts_2d, pts_3d, intrinsic_matrix, dist_coeffs)
    else:
        intrinsic_matrix, dist_coeffs, extrinsic_translation, extrinsic_rotation, reprojection_error = \
            calibrate_full(pts_2d, pts_3d, rec_img_size)

    extrinsic_matrix = convert_extrinsic_vectors_to_matrix(extrinsic_translation, extrinsic_rotation)

    return intrinsic_matrix, dist_coeffs, extrinsic_matrix, reprojection_error

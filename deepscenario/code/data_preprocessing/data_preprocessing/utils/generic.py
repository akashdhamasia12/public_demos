import json
import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def load_json(path_to_json: str) -> dict:
    with open(path_to_json) as file:
        return json.load(file)


def save_json(data: dict, path_to_json: str) -> None:
    with open(path_to_json, 'w') as file:
        json.dump(data, file, indent=4)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)



def convert_extrinsic_vectors_to_matrix(extrinsic_translation: np.ndarray,
                                        extrinsic_rotation: np.ndarray) -> np.ndarray:
    return np.column_stack([Rotation.from_rotvec(extrinsic_rotation).as_matrix(), extrinsic_translation])


def convert_extrinsic_matrix_to_vectors(extrinsic_matrix: np.ndarray) -> tuple:
    return np.squeeze(extrinsic_matrix[:, 3]), Rotation.from_matrix(extrinsic_matrix[:3, :3]).as_rotvec()


def convert_extrinsic_matrix_to_camera_transform(extrinsic_matrix: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Camera_resectioning (section: Extrinsic parameters)
    R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = R.T
    camera_transform[:3, 3] = -R.T.dot(t)
    return camera_transform



def invert_camera_matrix(intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
    camera_matrix_inv = np.zeros_like(extrinsic_matrix)
    camera_matrix_inv[:3, :3] = np.dot(extrinsic_matrix[:3, :3].T, np.linalg.inv(intrinsic_matrix))
    camera_matrix_inv[:3, 3] = np.dot(-extrinsic_matrix[:3, :3].T, extrinsic_matrix[:3, 3])
    return camera_matrix_inv


def project_to_image(pts_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate([pts_3d, np.ones((len(pts_3d), 1))], axis=1)
    pts_2d = np.dot(camera_matrix, pts_3d_homo.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def unproject_from_image(pts_2d: np.ndarray, depth: np.ndarray, camera_matrix_inv: np.ndarray) -> np.ndarray:
    # pts_2d: n x 2
    # depth: n x 1
    # camera_matrix_inv: 3 x 4
    # return: n x 3
    pts_2d_homo = np.concatenate([pts_2d, np.ones((pts_2d.shape[0], 1))], axis=1)
    pts_3d_homo = np.concatenate([depth * pts_2d_homo, np.ones((pts_2d_homo.shape[0], 1))], axis=1)
    pts_3d = np.dot(camera_matrix_inv, pts_3d_homo.T).T[:, :3]
    return pts_3d


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Converts an array of angles to the interval [-pi, pi)"""
    return np.mod(angles + np.pi, 2 * np.pi) - np.pi


def undistort_image(img: np.ndarray, intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(intrinsic_matrix, dist_coeffs, None, intrinsic_matrix, (w, h),
                                             cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_AREA)


def resize_image(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return img


def normalize_image_coordinates(pts_2d: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    size = max(img_width, img_height)
    pts_2d_norm = np.empty((len(pts_2d), 2))
    pts_2d_norm[:, 0] = (pts_2d[:, 0] + 0.5 - img_width / 2.0) / size
    pts_2d_norm[:, 1] = (pts_2d[:, 1] + 0.5 - img_height / 2.0) / size
    return pts_2d_norm


def denormalize_image_coordinates(pts_2d_norm: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    size = max(img_width, img_height)
    pts_2d = np.empty((len(pts_2d_norm), 2))
    pts_2d[:, 0] = pts_2d_norm[:, 0] * size - 0.5 + img_width / 2.0
    pts_2d[:, 1] = pts_2d_norm[:, 1] * size - 0.5 + img_height / 2.0
    return pts_2d

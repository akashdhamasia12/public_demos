import numpy as np
import cv2
import open3d as o3d

from data_preprocessing.utils.generic import convert_extrinsic_matrix_to_camera_transform, invert_camera_matrix, \
    unproject_from_image


def raycast_features(pts_2d: np.ndarray, dist_coeffs: np.ndarray, intrinsic_matrix: np.ndarray,
                     extrinsic_matrix: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> tuple:
    pts_2d = cv2.undistortPoints(pts_2d, intrinsic_matrix, dist_coeffs, P=intrinsic_matrix).reshape(-1, 2)
    camera_matrix_inv = invert_camera_matrix(intrinsic_matrix, extrinsic_matrix)
    camera_transform = convert_extrinsic_matrix_to_camera_transform(extrinsic_matrix)
    camera_position = camera_transform[:3, 3]

    # compute origin [ox, oy, oz] and direction [dx, dy, dz] of rays
    ray_origins = np.stack([camera_position] * pts_2d.shape[0])
    ray_directions = -camera_position + unproject_from_image(pts_2d, 1, camera_matrix_inv)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.concatenate([ray_origins, ray_directions], axis=-1).astype(np.float32)
    distances = scene.cast_rays(rays)['t_hit'].numpy()

    idxs_hit = np.where(~np.isinf(distances))[0]  # unhit points have infinite hit distance
    pts_3d = np.full((len(ray_origins), 3), np.inf)
    pts_3d[idxs_hit] = ray_origins[idxs_hit] + ray_directions[idxs_hit] * distances[idxs_hit, np.newaxis]
    return pts_3d, idxs_hit


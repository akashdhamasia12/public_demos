import numpy as np
from dataclasses import fields
import yaml
import pandas as pd

from data_preprocessing.utils.generic import load_json, convert_extrinsic_vectors_to_matrix
from data_preprocessing.utils.features import FeatureConfig


def load_feature_config(path_to_config: str) -> FeatureConfig:
    feature_config = FeatureConfig()

    with open(path_to_config, 'r') as file:
        config_loaded = yaml.safe_load(file)

    feature_config_attributes = [field.name for field in fields(feature_config)]
    for key, value in config_loaded.items():
        if key in feature_config_attributes:
            setattr(feature_config, key, value)

    return feature_config


def load_reconstruction(path_to_reconstruction: str) -> dict:
    return load_json(path_to_reconstruction)[0]


def load_reference_utm(path_to_georeference: str) -> dict:
    with open(path_to_georeference) as file:
        _, _, utm_zone = file.readline().split()
        zone_number = utm_zone[:-1]
        hemisphere = utm_zone[-1]
        x, y = file.readline().split()
    return dict(zone=(int(zone_number), hemisphere), offset=[float(x), float(y), 0.])


def filter_tracks(tracks: pd.DataFrame, img_fname: str, point_ids: list) -> pd.DataFrame:
    tracks_filtered = tracks[tracks.img_fname.eq(img_fname)]  # filter with respect to img_fname
    tracks_filtered = tracks_filtered[tracks_filtered.track_id.isin(point_ids)]  # filter with respect to point_ids
    return tracks_filtered


def get_exif_fname(img_fname: str) -> str:
    return img_fname + '.exif'


def get_features_fname(img_fname: str) -> str:
    return img_fname + '.features.npz'


def get_intrinsic_parameters(cameras: dict):
    assert len(cameras.keys()) == 1, 'More than one camera defined'
    camera_id = next(iter(cameras))
    assert cameras[camera_id]['projection_type'] == 'brown'

    # intrinsic matrix (https://www.opensfm.org/docs/_modules/opensfm/types.html#BrownPerspectiveCamera)
    intrinsic_matrix_normalized = np.array([[cameras[camera_id]['focal_x'], 0., cameras[camera_id]['c_x']],
                                            [0., cameras[camera_id]['focal_y'], cameras[camera_id]['c_y']],
                                            [0., 0., 1.]])

    w = cameras[camera_id]['width']
    h = cameras[camera_id]['height']
    s = max(w, h)
    normalized_to_pixel = np.array([[s, 0, w / 2.0],
                                    [0, s, h / 2.0],
                                    [0, 0, 1]])
    intrinsic_matrix = np.dot(normalized_to_pixel, intrinsic_matrix_normalized)

    # distortion coefficients
    dist_coeffs = np.array([cameras[camera_id]['k1'], cameras[camera_id]['k2'], cameras[camera_id]['p1'],
                            cameras[camera_id]['p2'], cameras[camera_id]['k3']])

    return intrinsic_matrix, dist_coeffs


def get_extrinsic_vectors(reconstruction: dict, shot_id: str) -> tuple:
    shot = reconstruction['shots'].get(shot_id)
    assert shot is not None, 'Given shot is not in reconstruction'
    return shot['translation'], shot['rotation']


def get_extrinsic_matrix(reconstruction: dict, shot_id: str) -> np.ndarray:
    extrinsic_translation, extrinsic_rotation = get_extrinsic_vectors(reconstruction, shot_id)
    return convert_extrinsic_vectors_to_matrix(extrinsic_translation, extrinsic_rotation)


def get_reconstructed_points(reconstruction: dict) -> np.ndarray:
    return np.array([reconstruction['points'][key]['coordinates'] for key in reconstruction['points'].keys()])

import os
import open3d as o3d

from data_preprocessing.utils.generic import load_json
from data_preprocessing.utils.opensfm import load_feature_config, load_reconstruction, get_intrinsic_parameters, \
    load_reference_utm


class ReferenceMapping:
    def __init__(self, mapping_dir: str, mapping_meta: dict) -> None:
        self.mapping_dir = mapping_dir
        self.mapping_meta = mapping_meta
        self.odm_dir = os.path.join(mapping_dir, 'odm')
        self.feature_config = load_feature_config(os.path.join(self.odm_dir, 'opensfm', 'config.yaml'))
        self.reconstruction = load_reconstruction(os.path.join(self.odm_dir, 'opensfm', 'reconstruction.json'))
        self.intrinsic_matrix, self.dist_coeffs = get_intrinsic_parameters(self.reconstruction['cameras'])
        self.mesh = o3d.io.read_triangle_mesh(os.path.join(self.odm_dir, 'odm_meshing', 'odm_mesh.ply'))
        self.reference_utm = load_reference_utm(os.path.join(self.odm_dir, 'odm_georeferencing',
                                                             'odm_georeferencing_model_geo.txt'))
        self.images_dir = os.path.join(self.odm_dir, 'images')

        self.keypoints = None if mapping_meta['keypoints'] is None else \
            load_json(os.path.join(mapping_dir, mapping_meta['keypoints']['file']))

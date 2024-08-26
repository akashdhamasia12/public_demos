import os
import numpy as np
import cv2
from tqdm import tqdm
from typing import Optional, Iterator, Tuple
from dataclasses import dataclass

from data_preprocessing.utils.generic import convert_extrinsic_vectors_to_matrix, project_to_image, undistort_image
from data_preprocessing.utils.multi_video_capture import MultiVideoCapture


def draw_keypoints(img: np.ndarray, pts_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    pts_2d = project_to_image(pts_3d, camera_matrix)
    for pt_2d in pts_2d:
        cv2.drawMarker(img, tuple(pt_2d.astype(np.int32)), color=(0, 0, 255), markerSize=12,
                       markerType=cv2.MARKER_CROSS, thickness=2)
    return img


@dataclass
class ImgPreprocessor:
    video_files: list
    intrinsic_matrix: np.ndarray
    dist_coeffs: np.ndarray
    keypoints: Optional[dict] = None

    def preprocess(self, img_dicts: list) -> Iterator[Tuple[np.ndarray, dict]]:
        mvideo_cap = MultiVideoCapture(self.video_files)

        for img_dict in img_dicts:
            extrinsic_matrix = convert_extrinsic_vectors_to_matrix(img_dict['extrinsic_translation'],
                                                                   img_dict['extrinsic_rotation'])
            camera_matrix = np.dot(self.intrinsic_matrix, extrinsic_matrix)

            success, img = mvideo_cap.get_frame(img_dict['frame_id'])
            assert success, f'Extracting frame {img_dict["frame_id"]} failed'

            # undistort image
            img = undistort_image(img, self.intrinsic_matrix, self.dist_coeffs)

            # draw keypoints
            if self.keypoints is not None:
                img = draw_keypoints(img, self.keypoints['coordinates'], camera_matrix)

            yield img, img_dict

        mvideo_cap.release()


def preprocess_images(video_files: list, intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray, img_dicts: list,
                      keypoints: Optional[dict], save_dir: str, quiet: bool = False) -> None:
    img_preprocessor = ImgPreprocessor(video_files, intrinsic_matrix, dist_coeffs, keypoints)
    for img, img_dict in tqdm(img_preprocessor.preprocess(img_dicts), desc='Preprocessing images', total=len(img_dicts),
                              disable=quiet):
        cv2.imwrite(os.path.join(save_dir, img_dict['file_name']), img)



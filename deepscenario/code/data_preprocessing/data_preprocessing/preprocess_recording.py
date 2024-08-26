import os
import logging
import argparse
import tempfile
import numpy as np
import PIL.Image
from tqdm import tqdm
import open3d as o3d

from data_preprocessing.utils.extract_frames import extract_frames
from data_preprocessing.utils.reference_mapping import ReferenceMapping
from data_preprocessing.utils.generic import load_json, save_json
from data_preprocessing.utils.features import FeatureConfig
from data_preprocessing.utils.localization import Localizer, filter_extrinsics
from data_preprocessing.utils.multi_video_capture import MultiVideoCapture
from data_preprocessing.metas.preprocessing_meta import PreprocessingMeta
from data_preprocessing.match_reference_images import match_reference_images_wrapper, plot_matching_pairs
from data_preprocessing.calibrate_camera import calibrate_camera
from data_preprocessing.preprocess_images import preprocess_images


def configure_logging(recording_dir: str) -> str:
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger()
    list(map(logger.removeHandler, logger.handlers))
    logging_prefix = f'[{os.path.basename(os.path.abspath(recording_dir))}]'
    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format=f'[%(levelname)s]{logging_prefix} %(message)s', level=log_level)
    return logging_prefix


def load_metas_and_directories(recording_dir: str, mappings_root: str) -> tuple:
    recording_meta = load_json(os.path.join(recording_dir, 'recording_meta.json'))
    ref_mapping_dir = os.path.join(mappings_root, recording_meta['reference_mapping_directory'])
    ref_mapping_meta = load_json(os.path.join(ref_mapping_dir, 'reference_mapping_meta.json'))
    return recording_meta, ref_mapping_meta, ref_mapping_dir


def get_calibration_items(recording_meta: dict) -> list:
    calib_fnames = recording_meta['calibration_file_names']
    intrinsic_matrix = recording_meta['intrinsic_matrix']
    intrinsic_matrix = np.array(intrinsic_matrix) if intrinsic_matrix is not None else None
    dist_coeffs = recording_meta['distortion_coefficients']
    dist_coeffs = np.array(dist_coeffs) if dist_coeffs is not None else None
    return calib_fnames, intrinsic_matrix, dist_coeffs


def extract_key_frame_from_recording(mvideo_cap: MultiVideoCapture, recording_meta: dict,
                                     require_exif: bool) -> PIL.Image.Image:
    key_frame_id = max(recording_meta['usable_frames'][0], mvideo_cap.frame_offsets[mvideo_cap.n_videos // 2])
    key_video_idx = mvideo_cap.get_video_index(key_frame_id)
    key_frame_offset = mvideo_cap.frame_offsets[key_video_idx]
    key_video_file = mvideo_cap.path_to_videos[key_video_idx]
    key_video_frame_id = key_frame_id - key_frame_offset

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_files = extract_frames(key_video_file, tmp_dir, total_frames=1, start_frame_id=key_video_frame_id,
                                   frame_dist=1, img_format='jpg', require_exif=require_exif, quiet=True)
        return PIL.Image.open(img_files[0])


def get_img_dicts_from_parameters(recording_meta: dict, frame_dist: int, total_frames: int, frame_count: int) -> list:
    usable_frames = recording_meta['usable_frames']
    start_frame_id = usable_frames[0]
    stop_frame_id = frame_count - 1 if total_frames is None else \
        min(start_frame_id + total_frames * frame_dist, frame_count) - 1
    stop_frame_id = min(stop_frame_id, usable_frames[1]) if usable_frames[1] != -1 else stop_frame_id
    return [{'frame_id': i, 'sequence_id': 0} for i in range(start_frame_id, stop_frame_id + 1, frame_dist)]


def get_img_fname(recording_meta: dict, frame_id: int, max_digits: int, img_ext: str = '.jpg') -> str:
    return recording_meta['create_date'] + 'T' + recording_meta['create_time'] + '_' + str(frame_id).zfill(
        max_digits) + img_ext


def localize_images(mvideo_cap: MultiVideoCapture, recording_meta: dict, total_frames: int, frame_dist: int,
                    localizer: Localizer, quiet: bool) -> list:
    max_digits = len(str(mvideo_cap.total_frame_count))
    img_dicts = get_img_dicts_from_parameters(recording_meta, frame_dist, total_frames, mvideo_cap.total_frame_count)

    for img_dict in tqdm(img_dicts, desc='Localizing images', disable=quiet):
        success, img = mvideo_cap.get_frame(img_dict['frame_id'])
        assert success, f'Extracting frame {img_dict["frame_id"]} failed'

        # localize camera
        try:
            extrinsic_trans, extrinsic_rot = localizer.localize(img)
        except Exception as err:
            error_msg = f'Localizer failed at frame id {img_dict["frame_id"]}: {err}'
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # update image dict
        img_dict['file_name'] = get_img_fname(recording_meta, img_dict['frame_id'], max_digits)
        img_dict['extrinsic_translation'] = extrinsic_trans
        img_dict['extrinsic_rotation'] = extrinsic_rot

    return img_dicts


def preprocess_recording(recording_dir: str, mappings_root: str, save_dir: str, total_frames: int = None,
                         frame_dist: int = 1, no_images: bool = False, no_mask: bool = False,
                         add_keypoints: bool = False, viz_matches: bool = False, viz_localization: bool = False,
                         quiet: bool = False) -> None:
    configure_logging(recording_dir)

    recording_meta, ref_mapping_meta, ref_mapping_dir = load_metas_and_directories(recording_dir, mappings_root)
    ref_mapping = ReferenceMapping(ref_mapping_dir, ref_mapping_meta)
    calib_fnames, intrinsic_matrix, dist_coeffs = get_calibration_items(recording_meta)

    # extract key frame
    recording_files = [os.path.join(recording_dir, fname) for fname in recording_meta['recording_file_names']]
    mvideo_cap = MultiVideoCapture(recording_files)
    key_frame = extract_key_frame_from_recording(mvideo_cap, recording_meta, len(calib_fnames) == 0)

    # match key frame with reference frames
    matching_pairs = match_reference_images_wrapper(key_frame, calib_fnames, ref_mapping, quiet)
    if not matching_pairs:
        error_msg = 'No matches found between start frame and reference frames'
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    if viz_matches:
        save_matches_dir = os.path.join(save_dir, 'matches')
        plot_matching_pairs(matching_pairs, save_matches_dir, quiet)

    # calibrate camera
    intrinsic_matrix, dist_coeffs, start_extrinsic_matrix, reprojection_error = \
        calibrate_camera(matching_pairs, intrinsic_matrix, dist_coeffs, key_frame.size)

    # localize images
    feature_config = FeatureConfig(feature_min_frames=1000, feature_process_size=key_frame.width // 2)
    localizer = Localizer.initialize_from_frame(key_frame, intrinsic_matrix, dist_coeffs, start_extrinsic_matrix,
                                                ref_mapping.mesh, feature_config)
    img_dicts = localize_images(mvideo_cap, recording_meta, total_frames, frame_dist, localizer, quiet)
    img_dicts = filter_extrinsics(img_dicts)

    # release video capture
    mvideo_cap.release()

    # create save directory
    os.makedirs(save_dir, exist_ok=True)

    # init and save preprocessing meta
    preprocessing_meta = vars(
        PreprocessingMeta(recording_dir, ref_mapping.reference_utm, intrinsic_matrix, dist_coeffs, reprojection_error,
                          recording_meta, frame_dist, img_dicts))
    path_to_preprocessing_meta = os.path.join(save_dir, 'preprocessing_meta.json')
    save_json(preprocessing_meta, path_to_preprocessing_meta)

    reprojection_error_limit = 4
    if reprojection_error > reprojection_error_limit:
        logging.warning(f'Reprojection error {reprojection_error:.1f} > {reprojection_error_limit}: check the' + \
                        'keypoints and consider using SIFT.')

    # preprocess images
    if not no_images:
        video_files = [os.path.join(recording_dir, fname) for fname in recording_meta['recording_file_names']]
        preprocess_images(video_files, intrinsic_matrix, dist_coeffs, img_dicts,
                          ref_mapping.keypoints if add_keypoints else None, save_dir, quiet=quiet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to preprocess the frames of a recording')
    parser.add_argument('--recording_dir', help='Directory where recording is stored', type=str, required=True)
    parser.add_argument('--mappings_root', help='Directory where all mappings are stored', type=str, required=True)
    parser.add_argument('--save_dir', help='Directory where the computed output is stored', type=str, required=True)
    parser.add_argument('--total', help='Number of frames to extract', dest='total_frames', type=int, required=False)
    parser.add_argument('--no_images', help='Do not extract the images from the recording', action='store_true')
    parser.add_argument('--add_keypoints', help='Add the keypoints to the images', action='store_true')
    parser.add_argument('--viz_matches', help='Visualize matches with reference frames', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_recording(**vars(args))

import numpy as np
import cv2
from typing import Tuple


class VideoCapture:
    def __init__(self, path_to_video: str) -> None:
        self.video_cap = cv2.VideoCapture(path_to_video)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, frame_id: int) -> Tuple[bool, np.ndarray]:
        frame_id_curr = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        assert frame_id >= frame_id_curr, 'Can only interate forward'

        # frame id outside of valid interval
        if frame_id < 0 or frame_id >= self.frame_count:
            return False, None

        # iterate video capture to respective frame
        # (note that OpenCV's video_cap.set() is not properly working with video_cap.get())
        while not int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame_id:
            _, _ = self.video_cap.read()

        return self.video_cap.read()

    def release(self) -> None:
        self.video_cap.release()

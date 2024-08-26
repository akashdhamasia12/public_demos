import os
import datetime
import exiftool


class VideoMeta:
    def __init__(self, path_to_video: str) -> None:
        with exiftool.ExifTool(common_args=['-G', '-n', '-api', 'largefilesupport=1']) as et:
            video_exif = et.get_metadata(path_to_video)

        self.file_name = video_exif['File:FileName']
        self.directory = video_exif['File:Directory']
        self.camera_manufacturer = video_exif['QuickTime:Make'] if 'QuickTime:Make' in video_exif else 'Unknown'
        self.camera_model = video_exif['QuickTime:Model'] if 'QuickTime:Model' in video_exif else 'Unknown'
        self.media_size = video_exif['QuickTime:MediaDataSize']
        self.create_datetime = datetime.datetime.strptime(video_exif['QuickTime:CreateDate'], '%Y:%m:%d %H:%M:%S')
        self.duration = video_exif['QuickTime:Duration']
        self.image_width = video_exif['QuickTime:ImageWidth']
        self.image_height = video_exif['QuickTime:ImageHeight']
        self.frame_rate = video_exif['QuickTime:VideoFrameRate']
        self.latitude = video_exif['Composite:GPSLatitude'] if 'Composite:GPSLatitude' in video_exif else None
        self.longitude = video_exif['Composite:GPSLongitude'] if 'Composite:GPSLongitude' in video_exif else None
        self.altitude = video_exif['Composite:GPSAltitude'] if 'Composite:GPSAltitude' in video_exif else None

    def get_path_to_video(self) -> str:
        return os.path.join(self.directory, self.file_name)

import os
import subprocess
from itertools import product
from os.path import dirname, join


def main(photos_dir: str, videos_dir: str):
    photos_paths = map(lambda x: join(dirname(__file__), photos_dir, x), os.listdir(photos_dir))
    videos_paths = map(lambda x: join(dirname(__file__), videos_dir, x), os.listdir(videos_dir))
    combinations = list(product(photos_paths, videos_paths))

    for photo_path, video_path in combinations:
        command = f'python3 inference.py "{photo_path}" "{video_path}"'
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    path_to_anton_photos = join('demo_file', 'anton_4', 'faces')
    path_to_anton_videos = join('demo_file', 'anton_4', 'videos')
    main(path_to_anton_photos, path_to_anton_videos)

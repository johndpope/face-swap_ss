import os
import shutil
import subprocess
from datetime import datetime
from glob import glob
from os.path import basename, dirname, exists, isfile, join, splitext

import cv2
import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

from util.reverse2original import reverse2wholeimage


def execute_command(command: str, error_message: str,
                    print_on_error: bool = False,
                    raise_on_error: bool = False) -> None:
    exitcode = subprocess.call(command, shell=True)
    if exitcode == 1:
        if print_on_error:
            print(error_message)
        if raise_on_error:
            raise Exception(error_message)
    return exitcode

def extract_audio(video_path: str, audio_path: str) -> None:
    print(f'=> Extracting audio from video: "{basename(video_path)}"...')
    command = f'ffmpeg -hide_banner -loglevel error -i "{video_path}" -vn -ar 44100 -ac 2 -ab 192K -f wav -y "{audio_path}"'
    execute_command(command, f'##### Unable to extract audio of {video_path} ({datetime.now()}). #####', print_on_error=True)

def insert_audio(save_path, audio_path):
    if isfile(audio_path):
        temp_path = splitext(save_path)[0] + "_TEMP" + splitext(save_path)[1]
        os.rename(save_path, temp_path)
        command = f'ffmpeg -hide_banner -loglevel warning -i "{temp_path}" -i "{audio_path}" -c:v libx264 -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -pix_fmt yuv420p "{save_path}" -y'
        execute_command(command, f'Error while creating the video from the video {temp_path} and audio from {audio_path} ({datetime.now()}).', 
                        raise_on_error=True)
        os.remove(temp_path)

def get_frames_n(video_path: str) -> int:
    def _manual_count(handler):
        frames_n = 0
        while True:
            status, _ = handler.read()
            if not status:
                break
            frames_n += 1
        return frames_n 

    cap = cv2.VideoCapture(video_path)
    frames_n = _manual_count(cap)
    cap.release()
    return frames_n


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_veсtor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224):
    if exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir)

    audio_path = join(temp_results_dir, splitext(basename(video_path))[0] + '.wav')
    extract_audio(video_path, audio_path)

    frame_count = get_frames_n(video_path)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    for frame_index in tqdm(range(frame_count)): 
        _, frame = video.read()
        detect_results = detect_model.get(frame,crop_size)

        if detect_results is not None:
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []

            for frame_align_crop in frame_align_crop_list:
                frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                swap_result = swap_model(None, frame_align_crop_tenor, id_veсtor, None, True)[0]
                swap_result_list.append(swap_result)
            reverse2wholeimage(swap_result_list, frame_mat_list, crop_size, frame, join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)))
        else:
            frame = frame.astype(np.uint8)
            cv2.imwrite(join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

    video.release()

    image_filenames = sorted(glob(join(temp_results_dir, '*.jpg')))

    os.makedirs(dirname(save_path), exist_ok=True)
    clips = ImageSequenceClip(image_filenames, fps=fps)
    clips.write_videofile(save_path)
    insert_audio(save_path, audio_path)
    shutil.rmtree(temp_results_dir)
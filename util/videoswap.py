import mimetypes
import os
import shutil
import subprocess
from datetime import datetime
from os.path import basename, dirname, exists, isfile, join, normpath, splitext
from time import gmtime, perf_counter, strftime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from util.reverse2original import reverse2wholeimage


def get_media_type(path: str) -> str:
    try:
        media_type = mimetypes.guess_type(path)[0].split('/')[0]
    except:
        raise Exception(f'Unable to parse media type of {path} ({datetime.now()}).')
    assert media_type in ['image', 'video'], f'Unable to recognize media type of {path} ({datetime.now()}).'
    return media_type


def timer(func_name):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            script_time_start = perf_counter()
            result = func(*args, **kwargs)
            elapsed = strftime("%H:%M:%S", gmtime(perf_counter() - script_time_start))
            print(f'> Time elapsed on `{func_name}`: {elapsed} ({datetime.now()}).')
            return result
        return wrapper
    return actual_decorator


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

@timer('Extracting audio')
def extract_audio(video_path: str, audio_path: str) -> None:
    print(f'=> Extracting audio from video "{basename(video_path)}"...')
    command = f'ffmpeg -hide_banner -loglevel error -i "{video_path}" -vn -ar 44100 -ac 2 -ab 192K -f wav -y "{audio_path}"'
    execute_command(command, f'> > > > > Unable to extract audio of {video_path} ({datetime.now()}).', print_on_error=True)

@timer('Creating video')
def create_video(save_path: str, audio_path: str, frames_path: str, fps: float) -> str:
    print(f'=> Creating video from frames at "{frames_path}"...')
    os.makedirs(dirname(save_path), exist_ok=True)
    if isfile(audio_path):
        command = f'ffmpeg -hide_banner -loglevel warning -pattern_type glob -v 8 -r "{fps}" -i "{normpath(frames_path)}/*.jpg" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -y "{save_path}"'
        execute_command(command, f'> > > > > Error while creating the video from the frames of {frames_path} and audio from {audio_path} ({datetime.now()}).', 
                        raise_on_error=True)
    else:
        command = f'ffmpeg -hide_banner -loglevel warning -pattern_type glob -v 8 -r "{fps}" -i "{normpath(frames_path)}/*.jpg" -c:v libx264 -pix_fmt yuv420p -y "{save_path}"'
        execute_command(command, f'> > > > > Error while creating the video from the frames of {frames_path} ({datetime.now()}).', 
                        raise_on_error=True)

@timer('Getting number of frames')
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

@timer('Lowering resolution')
def lower_video_resolution(video_path: str) -> None:
    M = 1080
    vidcap = cv2.VideoCapture(video_path)
    width, height = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH), vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if width > M and height > M:
        print(f'=> Lowering resolution of the video "{basename(video_path)}" (smallest side is {M})...')
        video_temp_path = splitext(video_path)[0] + '_TEMP' + splitext(video_path)[1]
        os.rename(video_path, video_temp_path)
        scale = f'-vf scale="-2:{M}"' if width > height else f'-vf scale="{M}:-2"'
        command = f'ffmpeg -hide_banner -loglevel error -i "{video_temp_path}" {scale} -y "{video_path}"'
        execute_command(command, f'Unable to lower the resolution of the {video_path} ({datetime.now()}).', raise_on_error=True)
        os.remove(video_temp_path)
    return video_path

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

@timer('Swapping Face in video')
def video_swap(video_path, source_latend_id, face_swap_model, face_detector, seg_model, sr_model, apply_sr, output_path, is_prod, temp_results_dir='./temp_results', crop_size=224):
    lower_video_resolution(video_path)
    print(f'=> Swapping face in "{video_path}"...')
    if exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir)

    audio_path = join(temp_results_dir, splitext(basename(video_path))[0] + '.wav')
    extract_audio(video_path, audio_path)

    frame_count = get_frames_n(video_path)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    for frame_index in tqdm(range(frame_count), disable=is_prod): 
        _, frame = video.read()
        swap_frame(source_latend_id, face_swap_model, face_detector, seg_model, sr_model, apply_sr, temp_results_dir, crop_size, frame_index, frame)

    video.release()
    create_video(output_path, audio_path, temp_results_dir, fps)
    shutil.rmtree(temp_results_dir)

@timer('Swapping Face in photo')
def photo_swap(photo_path, source_latend_id, face_swap_model, face_detector, seg_model, sr_model, apply_sr, output_path, is_prod, temp_results_dir='./temp_results', crop_size=224):
    # lower_photo_resolution(photo_path)
    print(f'=> Swapping face in "{photo_path}"...')
    if exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir)

    photo = cv2.imread(photo_path)
    swap_frame(source_latend_id, face_swap_model, face_detector, seg_model, sr_model, apply_sr, temp_results_dir, crop_size, 0, photo, ext=splitext(output_path)[1])
    photo_infer_path = os.listdir(temp_results_dir)[0]
    os.makedirs(dirname(output_path), exist_ok=True)
    os.rename(join(temp_results_dir, photo_infer_path), output_path)
    shutil.rmtree(temp_results_dir)


def swap_frame(source_latend_id, face_swap_model, face_detector, seg_model, sr_model, apply_sr, temp_results_dir, crop_size, frame_index, frame, ext='.jpg'):
    with torch.no_grad():
        detect_results = face_detector.get(frame, crop_size)

        if not detect_results in [None, (None, None)]:
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []

            for frame_align_crop in frame_align_crop_list:
                frame_align_crop_tensor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                swap_result = face_swap_model(None, frame_align_crop_tensor, source_latend_id, None, True)[0]
                swap_result_list.append(swap_result)
            reverse2wholeimage(swap_result_list, frame_mat_list, crop_size, frame, seg_model, sr_model, apply_sr, 
                                join(temp_results_dir, f'frame_{frame_index:0>7d}{ext}'))
        else:
            frame = frame.astype(np.uint8)
            cv2.imwrite(join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

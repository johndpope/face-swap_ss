import sys
from os.path import basename, isfile, join, splitext
from shutil import copy2

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from insightface_func.face_detect_crop_single import Face_detect_crop
from models.models import create_model
from options.test_options import TestOptions
from util.videoswap import video_swap

model, app = None, None
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def initialize():
    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f')  # dummy arg to avoid bug
    opt = opt.parse()
    opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False
    torch.nn.Module.dump_patches = True
    global model
    model = create_model(opt)
    model.eval()
    global app
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(256, 256))


def infer(source, target, result_dir='./output', crop_size=224):
    assert isfile(source), f'Can\'t find source at {source}'
    assert isfile(target), f'Can\'t find target at {target}'
    output_filename = f'infer-{splitext(basename(source))[0]}-{splitext(basename(target))[0]}.mp4'
    output_path = join(result_dir, output_filename)

    assert model is not None
    assert app is not None

    img_a_whole = cv2.imread(source)
    img_a_align_crop, _ = app.get(img_a_whole, crop_size)
    if img_a_align_crop is None:
        copy2(target, output_path)
        return output_path
    img_a_align_crop_pil = Image.fromarray(
        cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
    img_id = img_id.cuda()

    img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
    latend_id = model.netArc(img_id_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
    latend_id = latend_id.to('cuda')

    video_swap(target, latend_id, model, app, output_path)
    return output_path


if __name__ == "__main__":
    assert len(sys.argv) == 3, 'Usage: python3 inference.py "path/to/source_image" "path/to/target_video"'
    initialize()
    infer(sys.argv[1], sys.argv[2])

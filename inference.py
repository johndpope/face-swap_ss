import sys
from os.path import basename, isfile, join, splitext
from shutil import copy2

import cv2
import easydict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from face_seg.nets.MobileNetV2_unet import MobileNetV2_unet
from fsr.models.SRGAN_model import SRGANModel
from insightface_func.face_detect_crop_single import Face_detect_crop
from models.models import create_model
from options.test_options import TestOptions
from util.videoswap import video_swap

model, app, seg_model, sr_model = None, None, None, None
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def initialize():
    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f')  # dummy arg to avoid bug
    opt = opt.parse()
    opt.Arc_path = './weights/arcface_checkpoint.tar'
    opt.isTrain = False
    torch.nn.Module.dump_patches = True
    global model
    model = create_model(opt)
    model.eval()
    global app
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(256, 256))
    global seg_model
    seg_model = MobileNetV2_unet(None).to('cuda')
    state_dict = torch.load('./face_seg/checkpoints/model.pt', map_location='cpu')
    seg_model.load_state_dict(state_dict)
    seg_model.eval()
    global sr_model
    args = easydict.EasyDict({
        'gpu_ids': None,
        'batch_size': 32,
        'lr_G': 1e-4,
        'weight_decay_G': 0,
        'beta1_G': 0.9,
        'beta2_G': 0.99,
        'lr_D': 1e-4,
        'weight_decay_D': 0,
        'beta1_D': 0.9,
        'beta2_D': 0.99,
        'lr_scheme': 'MultiStepLR',
        'niter': 100000,
        'warmup_iter': -1,
        'lr_steps': [50000],
        'lr_gamma': 0.5,
        'pixel_criterion': 'l1',
        'pixel_weight': 1e-2,
        'feature_criterion': 'l1',
        'feature_weight': 1,
        'gan_type': 'ragan',
        'gan_weight': 5e-3,
        'D_update_ratio': 1,
        'D_init_iters': 0,
        'print_freq': 100,
        'val_freq': 1000,
        'save_freq': 10000,
        'crop_size': 0.85,
        'lr_size': 128,
        'hr_size': 512,
        'which_model_G': 'RRDBNet',
        'G_in_nc': 3,
        'out_nc': 3,
        'G_nf': 64,
        'nb': 16,
        'which_model_D': 'discriminator_vgg_128',
        'D_in_nc': 3,
        'D_nf': 64,
        'pretrain_model_G': 'weights/90000_G.pth',
        'pretrain_model_D': None
    })
    sr_model = SRGANModel(args, is_train=False)
    sr_model.load()
    sr_model.netG.to('cuda')
    sr_model.netG.eval();


def infer(source, target, apply_sr, result_dir='./output', crop_size=224):
    print(apply_sr)
    assert isfile(source), f'Can\'t find source at {source}'
    assert isfile(target), f'Can\'t find target at {target}'
    output_filename = f'infer-{splitext(basename(source))[0]}-{splitext(basename(target))[0]}{splitext(basename(target))[1]}'
    output_path = join(result_dir, output_filename)

    assert not model is None
    assert not app is None
    assert not seg_model is None
    assert not sr_model is None

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

    video_swap(target, latend_id, model, app, seg_model, sr_model, apply_sr, output_path)
    return output_path


if __name__ == "__main__":
    assert len(sys.argv) in [3, 4], 'Usage: python3 inference.py "path/to/source_image" "path/to/target_video" [--no_sr]'
    initialize()
    infer(sys.argv[1], sys.argv[2], not '--no_sr' in sys.argv)

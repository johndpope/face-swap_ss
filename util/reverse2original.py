import cv2
import easydict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from fsr.models.SRGAN_model import SRGANModel

esrgan_fsr_transform = transforms.Compose([transforms.Resize((128, 128)),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])

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

    # network G
    'which_model_G': 'RRDBNet',
    'G_in_nc': 3,
    'out_nc': 3,
    'G_nf': 64,
    'nb': 16,

    # network D
    'which_model_D': 'discriminator_vgg_128',
    'D_in_nc': 3,
    'D_nf': 64,

    # data dir
    'pretrain_model_G': 'weights/90000_G.pth',
    'pretrain_model_D': None
})


esrgan_fsr_model = SRGANModel(args, is_train=False)
esrgan_fsr_model.load()
esrgan_fsr_model.netG.to('cuda')
esrgan_fsr_model.netG.eval();

# from BasicSR.basicsr.archs.rrdbnet_arch import RRDBNet
# import torch

# esrgan_basicsr_path = 'BasicSR/experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'

# esrgan_basicsr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
# esrgan_basicsr_model.load_state_dict(torch.load(esrgan_basicsr_path)['params'], strict=True)
# esrgan_basicsr_model.eval()
# esrgan_basicsr_model = esrgan_basicsr_model.to('cuda')

def reverse2wholeimage(swaped_imgs, mats, crop_size, oriimg, seg_model, save_path=''):
    target_image_list = []
    img_mask_list = []
    for swaped_img, mat in zip(swaped_imgs, mats):

        # https://github.com/kampta/face-seg
        seg_mask_logits = seg_model(swaped_img.unsqueeze(0))
        seg_mask = seg_mask_logits.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
        seg_mask = np.argmax(seg_mask, axis=2) == 1
        img_mask = np.array(seg_mask * 255, dtype=float)
        # img_mask = np.full((crop_size, crop_size), 255, dtype=float)

        # SR-ESRGAN_fsr https://github.com/ewrfcas/Face-Super-Resolution
        swaped_img = esrgan_fsr_transform(torch.clone(swaped_img))
        swaped_img = esrgan_fsr_model.netG(swaped_img.unsqueeze(0))
        swaped_img = swaped_img.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))
        swaped_img = swaped_img / 2.0 + 0.5

        # # SR-ESRGAN_BasicSR https://github.com/xinntao/BasicSR/blob/master/inference/inference_esrgan.py
        # swaped_img = esrgan_basicsr_model(swaped_img.unsqueeze(0))
        # swaped_img = swaped_img.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))

        # swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))

        mat_rev = cv2.invertAffineTransform(mat)
        mat_rev_face = np.array(mat_rev)
        mat_rev_face[:2, :2] = mat_rev_face[:2, :2] / (swaped_img.shape[0] / crop_size)

        orisize = (oriimg.shape[1], oriimg.shape[0])
        target_image = cv2.warpAffine(swaped_img, mat_rev_face, orisize)
        img_mask = cv2.warpAffine(img_mask, mat_rev, orisize)
        img_mask[img_mask > 20] = 255

        kernel = np.ones((10, 10), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)

        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255

        img_mask_list.append(img_mask)
        target_image_list.append(target_image)

    img = np.array(oriimg, dtype=np.float)
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        img = img_mask * target_image + (1-img_mask) * img

    final_img = img.astype(np.uint8)
    cv2.imwrite(save_path, final_img)

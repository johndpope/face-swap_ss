import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

esrgan_fsr_transform = transforms.Compose([transforms.Resize((128, 128)),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])


def reverse2wholeimage(swaped_imgs, mats, crop_size, oriimg, seg_model, sr_model, save_path=''):
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
        swaped_img = sr_model.netG(swaped_img.unsqueeze(0))
        swaped_img = swaped_img.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))
        swaped_img = np.clip(swaped_img / 2.0 + 0.5, 0, 1)
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

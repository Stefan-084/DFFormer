import os
from os import path as osp

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def image_align(deblurred, gt):
    zs = (np.sum(gt * deblurred) / np.sum(deblurred * deblurred)) * deblurred
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0)

    _, warp_matrix = cv2.findTransformECC(
        cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
        warp_matrix,
        warp_mode,
        criteria,
        inputMask=None,
        gaussFiltSize=5,
    )

    target_shape = gt.shape
    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype="float32"),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return zr * cr, gt * cr, cr


def masked_psnr(image_true, image_test, image_mask, data_range=1.0):
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range**2) / err)


def masked_ssim(tar_img, prd_img, mask):
    _, ssim_map = structural_similarity(
        tar_img,
        prd_img,
        channel_axis=2,
        gaussian_weights=True,
        use_sample_covariance=False,
        data_range=1.0,
        full=True,
    )
    ssim_map = ssim_map * mask
    radius = int(3.5 * 1.5 + 0.5)
    win_size = 2 * radius + 1
    pad = (win_size - 1) // 2
    ssim_map = ssim_map[pad:-pad, pad:-pad, :]
    mask = mask[pad:-pad, pad:-pad, :]
    ssim_map = ssim_map.sum(axis=0).sum(axis=0) / mask.sum(axis=0).sum(axis=0)
    return float(np.mean(ssim_map))


def compute_realblur_metrics(pred_img, gt_img):
    try:
        pred_img, gt_img, mask = image_align(pred_img, gt_img)
    except cv2.error:
        mask = np.ones_like(pred_img, dtype=np.float32)
    psnr = masked_psnr(gt_img, pred_img, mask)
    ssim = masked_ssim(gt_img, pred_img, mask)
    return psnr, ssim


def save_rgb_image(path, image):
    os.makedirs(osp.dirname(path), exist_ok=True)
    image = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

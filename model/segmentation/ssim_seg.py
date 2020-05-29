from skimage.measure import compare_ssim
import cv2
import numpy as np


def ssim_seg(ori_img, re_img, win_size=11, threshold=0.1, gaussian_weights=False):
    """
    input:
    threhold:
    return: s_map: mask
    """
    # convert the images to grayscale
    if len(ori_img.shape) == 3:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    if len(re_img.shape) == 3:
        re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)

    # compute ssim , s: The value of ssim, d: the similar map
    mssim, ssim_map = compare_ssim(ori_img, re_img,  win_size=win_size, full=True, gaussian_weights=gaussian_weights)
    residual_map = 1.0 - np.clip(ssim_map, 0, 1)

    # get mask
    mask = np.zeros(residual_map.shape, dtype=np.uint8)
    mask[residual_map < threshold] = 0
    mask[residual_map >= threshold] = 255

    return mask


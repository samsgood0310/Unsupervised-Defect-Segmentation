import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def create_window(window_size, sigma, channel):
    """Generate gauss window
    """
    center = int(window_size / 2)
    _1D_gauss = torch.Tensor([exp((-(x - center) ** 2) / float(2 * (sigma ** 2))) for x in range(window_size)])
    _1D_gauss = _1D_gauss / _1D_gauss.sum()
    _1D_gauss= _1D_gauss.unsqueeze(1)
    # matrix multiply
    _2D_gauss = _1D_gauss.mm(_1D_gauss.t()).float().unsqueeze(0).unsqueeze(0)
    # kernel shape[out_channel, inchannel/groups, kernel_H, kernel_W]
    window = _2D_gauss.expand(channel, 1, window_size, window_size).contiguous()

    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # hyper-parameters
    C1 = 0.01**2
    C2 = 0.03**2
    padding = int(window_size / 2)

    # SSIM Map
    mean1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mean2 = F.conv2d(img2, window, padding=padding, groups=channel)
    mean1_sq = mean1.pow(2)
    mean2_sq = mean2.pow(2)
    mean1_2 = mean1*mean2

    sigma1_sq = F.conv2d(img1*img1, window, padding=padding, groups=channel) - mean1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=padding, groups=channel) - mean2_sq
    sigma1_2 = F.conv2d(img1*img2, window, padding=padding, groups=channel) - mean1_2

    ssim_map = ((2*mean1_2 + C1)*(2*sigma1_2 + C2))/((mean1_sq + mean2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


class SSIM_loss(nn.Module):
    def __init__(self, window_size, channel, is_cuda=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size[0]
        self.channel = channel
        self.window = create_window(window_size=self.window_size, sigma=1.5, channel=self.channel)
        if is_cuda is True:
            self.window = self.window.cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        loss = -_ssim(img1, img2, self.window, self.window_size, channel, True)

        return loss

import numpy as np
import scipy
import torch

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

import scipy.signal


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


# def rgb_ssim(
#     img0,
#     img1,
#     max_val,
#     filter_size=11,
#     filter_sigma=1.5,
#     k1=0.01,
#     k2=0.03,
#     return_map=False,
# ):
#     # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
#     assert len(img0.shape) == 3
#     assert img0.shape[-1] == 3
#     assert img0.shape == img1.shape

#     # Construct a 1D Gaussian blur filter.
#     hw = filter_size // 2
#     shift = (2 * hw - filter_size + 1) / 2
#     f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
#     filt = np.exp(-0.5 * f_i)
#     filt /= np.sum(filt)

#     # Blur in x and y (faster than the 2D convolution).
#     def convolve2d(z, f):
#         return scipy.signal.convolve2d(z, f, mode="valid")

#     filt_fn = lambda z: np.stack(
#         [
#             convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
#             for i in range(z.shape[-1])
#         ],
#         -1,
#     )
#     mu0 = filt_fn(img0)
#     mu1 = filt_fn(img1)
#     mu00 = mu0 * mu0
#     mu11 = mu1 * mu1
#     mu01 = mu0 * mu1
#     sigma00 = filt_fn(img0**2) - mu00
#     sigma11 = filt_fn(img1**2) - mu11
#     sigma01 = filt_fn(img0 * img1) - mu01

#     # Clip the variances and covariances to valid values.
#     # Variance must be non-negative:
#     sigma00 = np.maximum(0.0, sigma00)
#     sigma11 = np.maximum(0.0, sigma11)
#     sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
#     c1 = (k1 * max_val) ** 2
#     c2 = (k2 * max_val) ** 2
#     numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
#     denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
#     ssim_map = numer / denom
#     ssim = np.mean(ssim_map)
#     return ssim_map if return_map else ssim


# def gaussian(filter_size, sigma):
#     gauss = torch.tensor([exp(-(x - filter_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(filter_size)], dtype=torch.float32)
#     return gauss / gauss.sum()

# def create_window(filter_size, channel):
#     _1D_window = gaussian(filter_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, filter_size, filter_size).contiguous())
#     return window

# def _ssim(img1, img2, window, filter_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=filter_size//2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=filter_size//2, groups=channel)
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=filter_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=filter_size//2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=filter_size//2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     return ssim_map.mean() if size_average else ssim_map

# def rgb_ssim(img0, img1, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_map=False):
#     assert img0.shape == img1.shape
#     assert len(img0.shape) == 4 and img0.shape[1] == 3  # Expected shape [batch, channels, height, width]

#     channel = img0.shape[1]
#     window = create_window(filter_size, channel).to(img0.device)

#     ssim_map = _ssim(img0, img1, window, filter_size, channel, size_average=not return_map)
#     ssim = ssim_map.mean()
#     return ssim_map if return_map else ssim


def gaussian(window_size, sigma):
    gauss = np.exp(-np.square(np.arange(window_size) - window_size // 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).reshape(-1, 1)
    _2D_window = _1D_window @ _1D_window.T
    window = np.expand_dims(np.expand_dims(_2D_window, axis=0), axis=0)
    return window

def rgb_ssim(img0, img1, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    assert img0.shape == img1.shape
    assert img0.ndim == 4 and img0.shape[1] == 3  # Expected shape [batch, channels, height, width]

    window = create_window(filter_size, filter_sigma)
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    def ssim(img1, img2, window, window_size):
        mu1 = scipy.signal.convolve2d(img1, window, mode='same', boundary='wrap')
        mu2 = scipy.signal.convolve2d(img2, window, mode='same', boundary='wrap')
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = scipy.signal.convolve2d(img1 ** 2, window, mode='same', boundary='wrap') - mu1_sq
        sigma2_sq = scipy.signal.convolve2d(img2 ** 2, window, mode='same', boundary='wrap') - mu2_sq
        sigma12 = scipy.signal.convolve2d(img1 * img2, window, mode='same', boundary='wrap') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    # Compute SSIM for each channel, then average
    ssim_values = [ssim(img0[:, i, :, :], img1[:, i, :, :], window, filter_size) for i in range(3)]
    return np.mean(ssim_values)


import numpy as np
from tools.image.processing import normalize as proc_normalize

def PSNR(I, K, normalize=True, max_pixel=None, mask=None):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    I, K: np.ndarray
        Respectively the original and the compressed image.
    Returns:
        psnr: float
            The PSNR value in decibels (dB).
    """
    if normalize:
        I = proc_normalize(I)
        K = proc_normalize(K)
        max_pixel = 1.0
    elif max_pixel is None:
        max_pixel = np.nanmax(I)
    #
    if mask is not None:
        I =np.where(mask, I, np.nan)
        K =np.where(mask, K, np.nan)
    #
    mse = np.nanmean((I - K) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SSIM(img1, img2, K1=0.01, K2=0.03, window_size=11, normalize=True, L=None, mask=None):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    img1, img2: np.ndarray
        Input images to compare.
    data_range: float
        The data range of the input images (i.e., the difference between the maximum and minimum possible values).
    K1, K2: float
        Constants to stabilize the division with weak denominator.
    window_size: int
        Size of the Gaussian window.
    L: float
        The dynamic range of the pixel values (default is data_range).
    Returns:
        ssim: float
            The SSIM value.
    """
    if normalize:
        img1 = proc_normalize(img1)
        img2 = proc_normalize(img2)
        L = 1.0
    elif L is None:
        L = np.nanmax([np.nanmax(img1), np.nanmax(img2)]) - np.nanmin([np.nanmin(img1), np.nanmin(img2)])
    #
    if mask is None:
        mask = np.ones_like(img1, dtype=bool)
    #
    if mask is not None:
        img1 =np.where(mask, img1, np.nan)
        img2 =np.where(mask, img2, np.nan)
    #

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = np.nanmean(img1)
    mu2 = np.nanmean(img2)
    sigma1_sq = np.nanvar(img1)
    sigma2_sq = np.nanvar(img2)
    sigma12 = np.cov(img1[mask].flatten(), img2[mask].flatten())[0, 1]

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim
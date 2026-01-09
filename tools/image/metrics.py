import numpy as np
from tools.image.processing import normalize

def PSNR(I, K):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    I, K: np.ndarray
        Respectively the original and the compressed image.
    Returns:
        psnr: float
            The PSNR value in decibels (dB).
    """
    if not 0 <= np.min(I) <= np.max(I) <= 1:
        I = normalize(I)
    if not 0 <= np.min(K) <= np.max(K) <= 1:
        K = normalize(K)
    #
    mse = np.mean((I - K) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(I)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SSIM(img1, img2, K1=0.01, K2=0.03, window_size=11, L=None):
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
    if not 0 <= np.min(img1) <= np.max(img1) <= 1:
        img1 = normalize(img1)
    if not 0 <= np.min(img2) <= np.max(img2) <= 1:
        img2 = normalize(img2)
    #
    if L is None:
        L = np.max([img1.max(), img2.max()]) - np.min([img1.min(), img2.min()])

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim
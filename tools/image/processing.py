import numpy as np

def normalize(im: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array to the range [0, 1].

    Parameters
    ----------
    im : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        Normalized image array.
    """
    min_val = np.min(im)
    max_val = np.max(im)
    im_norm = (im - min_val) / (max_val - min_val)
    return im_norm
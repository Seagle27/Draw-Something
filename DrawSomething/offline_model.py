import numpy as np
from DrawSomething.utils import color_spaces


def compute_skin_probability(rg_pixels, skin_gmm, non_skin_gmm):
    """
    Given Nx2 array of (r, g) points, return the Nx1 array of P(skin|c)
    using the formula:
        P(skin|c) = p(c|skin) / [ p(c|skin) + p(c|non-skin) ]
    where p(c|skin) is from 'skin_gmm.score_samples', etc.
    """
    # score_samples returns log probabilities, so we exponentiate.
    # p(c|skin) = exp(score_samples(c))

    log_p_skin = skin_gmm.score_samples(rg_pixels)  # shape (N,)
    log_p_non_skin = non_skin_gmm.score_samples(rg_pixels)

    p_skin = np.exp(log_p_skin)
    p_non_skin = np.exp(log_p_non_skin)

    return p_skin / (p_skin + p_non_skin + 1e-8)


def compute_offline_mask(frame_bgr, skin_gmm, non_skin_gmm, threshold=0.5):
    """
    Given a BGR frame, compute skin segmentation using the offline GMM approach.
    Return a binary mask (H x W) of 0/255 indicating non-skin/skin.
    """
    h, w = frame_bgr.shape[:2]

    # Convert to (r,g) per pixel
    rg_frame = color_spaces.to_rg_space(frame_bgr)

    # Flatten for GMM input: (H*W, 2)
    flat_rg = rg_frame.reshape(-1, 2)

    # Compute skin probability
    p_skin = compute_skin_probability(flat_rg, skin_gmm, non_skin_gmm)

    # Threshold
    skin_mask = (p_skin > threshold).astype(np.uint8) * 255

    # Reshape to original frame size
    skin_mask = skin_mask.reshape(h, w)
    return skin_mask


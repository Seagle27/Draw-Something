import numpy as np
from DrawSomething import constants


def initialize_histograms():
    """Return two 3D hist arrays for skin and non-skin."""
    skin_h = np.zeros((constants.H_BINS, constants.S_BINS, constants.V_BINS), dtype=np.float32)
    non_skin_h = np.zeros((constants.H_BINS, constants.S_BINS, constants.V_BINS), dtype=np.float32)
    return skin_h, non_skin_h


def bin_indices_hsv(frame_hsv):
    """
    Vectorized binning: given an HSV image [H,W,3],
    return h_bin, s_bin, v_bin (each [H,W]) based on our bin sizes.
    """
    # Convert to int
    H_vals = frame_hsv[..., 0].astype(np.int32)  # range approx 0..180
    S_vals = frame_hsv[..., 1].astype(np.int32)  # range 0..255
    V_vals = frame_hsv[..., 2].astype(np.int32)  # range 0..255

    h_bin = (H_vals * constants.H_BINS // 181).clip(0, constants.H_BINS - 1)
    s_bin = (S_vals * constants.S_BINS // 256).clip(0, constants.S_BINS - 1)
    v_bin = (V_vals * constants.V_BINS // 256).clip(0, constants.V_BINS - 1)
    return h_bin, s_bin, v_bin


def hist_update_vectorized(hist, frame_hsv, mask=None):
    """
    Update 'hist' by counting (h_bin, s_bin, v_bin) over the region where mask=1.
    Vectorized using np.bincount.
    """
    # Flatten:
    h_bin, s_bin, v_bin = bin_indices_hsv(frame_hsv)
    if mask is not None:
        # we'll keep only the pixels where mask=1
        idx = (mask > 0)
        h_idx = h_bin[idx]
        s_idx = s_bin[idx]
        v_idx = v_bin[idx]
    else:
        h_idx = h_bin.ravel()
        s_idx = s_bin.ravel()
        v_idx = v_bin.ravel()

    # Combine into a single index for 3D histogram
    bin_idx = (h_idx * (constants.S_BINS * constants.V_BINS)
               + s_idx * constants.V_BINS
               + v_idx)

    # Count frequencies
    counts = np.bincount(bin_idx, minlength=constants.H_BINS * constants.S_BINS * constants.V_BINS)
    counts_3d = counts.reshape((constants.H_BINS, constants.S_BINS, constants.V_BINS))

    # Add to the existing histogram
    hist += counts_3d
    return hist


def build_ratio_lut(skin_hist, non_skin_hist):
    """
    Build a ratio Look-Up Table (LUT):
       ratio_lut[h,s,v] = (skin_hist[h,s,v]/sum_skin) / (non_skin_hist[h,s,v]/sum_non)
    We'll add eps to avoid zero division.
    """
    eps = 1e-8
    sum_skin = skin_hist.sum() + eps
    sum_non_skin = non_skin_hist.sum() + eps

    p_skin = skin_hist / sum_skin
    p_non_skin = non_skin_hist / sum_non_skin

    ratio_lut = (p_skin + eps) / (p_non_skin + eps)
    return ratio_lut


def compute_online_mask(frame_hsv, ratio_lut, threshold_hist=1.5):
    """
    Build a classification mask from ratio_lut.  The ratio_lut
    is shape [H_BINS, S_BINS, V_BINS], built from the current
    'online' histograms.
    """
    h_vals, s_vals, v_vals = bin_indices_hsv(frame_hsv)

    ratio_map = ratio_lut[h_vals, s_vals, v_vals]
    # Threshold => mask
    online_mask = (ratio_map > threshold_hist).astype(np.uint8) * 255
    return online_mask


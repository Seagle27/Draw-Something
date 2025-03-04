import numpy as np
import joblib
from DrawSomething.utils import color_spaces
from DrawSomething import constants


class OfflineModel:
    def __init__(self, threshold, skin_gmm_path=constants.SKIN_GMM, non_skin_gmm_path=constants.NON_SKIN_GMM, lut_size=256):
        self.skin_gmm_model = joblib.load(skin_gmm_path)
        self.non_skin_gmm_model = joblib.load(non_skin_gmm_path)
        self.prob_lut = self.build_rg_probability_lut(lut_size)
        self.threshold = threshold

    def compute_skin_probability(self, rg_pixels):
        """
        Given Nx2 array of (r, g) points, return the Nx1 array of P(skin|c)
        using the formula:
            P(skin|c) = p(c|skin) / [ p(c|skin) + p(c|non-skin) ]
        where p(c|skin) is from 'skin_gmm.score_samples', etc.

        This function in no longer used - as we are using the LUT instead.
        """

        log_p_skin = self.skin_gmm_model.score_samples(rg_pixels)  # shape (N,)
        log_p_non_skin = self.non_skin_gmm_model.score_samples(rg_pixels)

        p_skin = np.exp(log_p_skin)
        p_non_skin = np.exp(log_p_non_skin)

        return p_skin / (p_skin + p_non_skin + 1e-8)

    def compute_offline_mask(self, frame):
        """
        Given a BGR frame, compute skin segmentation using the offline GMM approach.
        Return a binary mask (H x W) of 0/255 indicating non-skin/skin.
        Look-Up Table (LUT) is used to speedup the process
        """
        h, w = frame.shape[:2]

        # Convert to (r,g) per pixel
        rg_frame = color_spaces.to_rg_space(frame)
        r = rg_frame[:, :, 0]
        g = rg_frame[:, :, 1]

        # Convert to LUT indices
        lut_size = self.prob_lut.shape[0]  # e.g. 256
        # We map r,g in [0..1] to [0..lut_size-1]
        i_indices = np.clip((r * (lut_size - 1)).astype(int), 0, lut_size - 1)
        j_indices = np.clip((g * (lut_size - 1)).astype(int), 0, lut_size - 1)

        # Lookup
        p_skin = self.prob_lut[i_indices, j_indices]  # shape [H,W]

        # Threshold
        skin_mask = (p_skin > self.threshold).astype(np.uint8) * 255

        # Reshape to original frame size
        skin_mask = skin_mask.reshape(h, w)
        return skin_mask, p_skin

    def build_rg_probability_lut(self, lut_size):
        """
        Build a 2D LUT for r,g in [0,1]. The LUT has shape [lut_size, lut_size].
        lut[i, j] = P(skin | r=i/(lut_size-1), g=j/(lut_size-1)).
        """
        # We'll create a grid of (r, g) pairs.
        rg_values = np.linspace(0.0, 1.0, lut_size)
        rr, gg = np.meshgrid(rg_values, rg_values, indexing='ij')

        # Flatten so we can call score_samples once
        flat_rg = np.column_stack([rr.ravel(), gg.ravel()])  # shape (lut_size^2, 2)

        log_p_skin = self.skin_gmm_model.score_samples(flat_rg)  # shape (N,)
        log_p_non = self.non_skin_gmm_model.score_samples(flat_rg)

        p_skin = np.exp(log_p_skin)
        p_non = np.exp(log_p_non)

        prob_skin = p_skin / (p_skin + p_non + 1e-8)

        # Reshape back to (lut_size, lut_size)
        lut = prob_skin.reshape(lut_size, lut_size)
        return lut


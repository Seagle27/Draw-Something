import numpy as np
import cv2
from DrawSomething import constants


class OnlineModel:
    def __init__(self, threshold, skin_mask, non_skin_mask, frame):
        self.skin_hist, self.non_skin_hist = self.create_new_histograms()
        self.threshold = threshold

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.init_histograms(frame_hsv, skin_mask, non_skin_mask)

    @staticmethod
    def create_new_histograms():
        """Return two 3D hist arrays for skin and non-skin."""
        skin_h = np.zeros((constants.H_BINS, constants.S_BINS, constants.V_BINS), dtype=np.float32)
        non_skin_h = np.zeros((constants.H_BINS, constants.S_BINS, constants.V_BINS), dtype=np.float32)
        return skin_h, non_skin_h

    def init_histograms(self, frame_hsv, skin_mask, non_skin_mask):
        self.skin_hist = self.hist_update_vectorized(self.skin_hist, frame_hsv, skin_mask)
        self.non_skin_hist = self.hist_update_vectorized(self.non_skin_hist, frame_hsv, non_skin_mask)

    @staticmethod
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

    def hist_update_vectorized(self, hist, frame_hsv, mask=None):
        """
        Update 'hist' by counting (h_bin, s_bin, v_bin) over the region where mask=1.
        Vectorized using np.bincount.
        """
        # Flatten:
        h_bin, s_bin, v_bin = self.bin_indices_hsv(frame_hsv)
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

    def build_ratio_lut(self):
        """
        Build a ratio Look-Up Table (LUT):
           ratio_lut[h,s,v] = (skin_hist[h,s,v]/sum_skin) / (non_skin_hist[h,s,v]/sum_non)
        We'll add eps to avoid zero division.
        """
        eps = 1e-8
        sum_skin = self.skin_hist.sum() + eps
        sum_non_skin = self.non_skin_hist.sum() + eps

        p_skin = self.skin_hist / sum_skin
        p_non_skin = self.non_skin_hist / sum_non_skin

        ratio_lut = (p_skin + eps) / (p_non_skin + eps)
        return ratio_lut

    def compute_online_mask(self, frame_hsv):
        """
        Build a classification mask from ratio_lut.  The ratio_lut
        is shape [H_BINS, S_BINS, V_BINS], built from the current
        'online' histograms.
        """
        h_vals, s_vals, v_vals = self.bin_indices_hsv(frame_hsv)
        ratio_lut = self.build_ratio_lut()
        ratio_map = ratio_lut[h_vals, s_vals, v_vals]
        # Threshold => mask
        online_mask = (ratio_map > self.threshold).astype(np.uint8) * 255
        return online_mask, ratio_map

    def update(self, frame_hsv, skin_mask, non_skin_mask):
        skin_new, non_skin_new = self.create_new_histograms()

        s_new = self.hist_update_vectorized(skin_new, frame_hsv, skin_mask)

        non_skin_mask[skin_mask > 0] = False
        n_new = self.hist_update_vectorized(non_skin_new, frame_hsv, non_skin_mask)

        self.skin_hist = 0.9 * self.skin_hist + 0.1 * s_new
        self.non_skin_hist = 0.9 * self.non_skin_hist + 0.1 * n_new


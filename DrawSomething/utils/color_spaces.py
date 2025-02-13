import numpy as np
import cv2

from DrawSomething import constants


def to_rg_space(frame, origin_color_space=constants.ColorSpace.BGR):
    """
    Vectorized conversion from BGR to (r,g).
    Returns:
      rg: shape [H, W, 2], float32 in [0..1].
    """
    if origin_color_space == constants.ColorSpace.BGR:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    elif origin_color_space == constants.ColorSpace.RGB:
        frame_rgb = frame
    else:
        raise ValueError(f"Origin Color space: {origin_color_space} isn't supported")

    # Sum across color channels
    denom = frame_rgb.sum(axis=2, keepdims=True) + 1e-6
    rg = frame_rgb / denom
    # Keep only r,g channels
    rg = rg[:, :, :2]
    return rg

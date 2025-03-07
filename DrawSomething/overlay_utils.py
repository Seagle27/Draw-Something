# overlay_utils.py

import numpy as np
import cv2

def apply_brush_mask(
    brush_png_path: str,
    main_img: np.ndarray,
    center_x: int,
    center_y: int,
    type: str = "black",
    invert: bool = True,
    threshold: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a brush mask from a PNG image onto the main image at (center_x, center_y).
    Returns (modified_main_img, brush_mask).
    """
    """
        Applies a brush mask from a PNG image onto a main image.

        The function:
          - Loads the brush PNG image in grayscale.
          - Thresholds it to create a binary mask.
          - Resizes the mask to 60x60.
          - Overlays the mask onto the main_img at the given (center_x, center_y)
            position such that the pixels corresponding to the mask's white area (255)
            are set to black in the main image.

        Parameters:
          brush_png_path (str): Path to the brush/eraser PNG image.
          main_img (numpy.ndarray): Main image (cv2 image) where the mask will be applied.
          center_x (int): X-coordinate of the center for mask placement.
          center_y (int): Y-coordinate of the center for mask placement.
          threshold (int): Threshold value for binarization (default is 128).

        Returns:
          numpy.ndarray: The modified main image with the mask applied.
        """
    mask_arr = np.zeros((45, 45), dtype=np.uint8)

    # 1. Load the brush image in grayscale
    brush_gray = cv2.imread(brush_png_path, cv2.IMREAD_GRAYSCALE)
    if brush_gray is None:
        raise ValueError(f"Could not load brush image from: {brush_png_path}")

    # 2. Create a binary mask by thresholding
    # Pixels >= threshold become 255, below become 0.
    _, mask = cv2.threshold(brush_gray, threshold, 255, cv2.THRESH_BINARY)
    mask_arr[:] = np.where(brush_gray > threshold, 255, 0).astype(np.uint8)

    # 3. Resize the mask to 60x60
    mask_60 = cv2.resize(mask, (45, 45), interpolation=cv2.INTER_AREA)

    # 4. Calculate the region of interest (ROI) in the main image
    mask_h, mask_w = mask_60.shape  # should be 60x60 - i changed to 45
    half_h, half_w = mask_h // 2, mask_w // 2

    # Compute the ROI coordinates while ensuring they stay within the main image bounds.
    y1 = max(0, center_y - half_h)
    y2 = min(main_img.shape[0], center_y + half_h)
    x1 = max(0, center_x - half_w)
    x2 = min(main_img.shape[1], center_x + half_w)

    # Adjust the mask region if the ROI is clipped (at the image boundaries)
    mask_y1 = half_h - (center_y - y1)
    mask_y2 = mask_y1 + (y2 - y1)
    mask_x1 = half_w - (center_x - x1)
    mask_x2 = mask_x1 + (x2 - x1)

    # 5. Apply the mask: set ROI pixels to black where mask value is 255.
    complementary_mask = cv2.bitwise_not(mask_60)
    roi = main_img[y1:y2, x1:x2]
    mask = mask_60
    if invert:
        mask = complementary_mask
    if type == "black":
        roi[mask[mask_y1:mask_y2, mask_x1:mask_x2] == 255] = (0, 0, 0)
    if type == "white":
        roi[mask[mask_y1:mask_y2, mask_x1:mask_x2] == 255] = (255, 255, 255)

    # 6. Return the modified image (still in cv2/NumPy format)
    return main_img, mask_arr

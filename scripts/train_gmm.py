import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.mixture import GaussianMixture
import joblib  # to save the trained models if desired


def to_rg_space(rgb_array):
    """
    Convert Nx3 array of RGB values into Nx2 array (r,g) in chromaticity space.
    Avoids division by zero by adding a small epsilon.
    """
    # rgb_array is shape (N, 3) -> columns [R, G, B]
    R = rgb_array[:, 0].astype(np.float32)
    G = rgb_array[:, 1].astype(np.float32)
    B = rgb_array[:, 2].astype(np.float32)

    denom = R + G + B + 1e-6  # small epsilon
    r = R / denom
    g = G / denom
    return np.column_stack((r, g))


# 1. Fetch the UCI Skin Segmentation dataset
skin_segmentation = fetch_ucirepo(id=229)

X = skin_segmentation.data.features  # (R, G, B)
y = skin_segmentation.data.targets  # skin=1, non-skin=2 (according to dataset docs)

# metadata
print(skin_segmentation.metadata)

# variable information
print(skin_segmentation.variables)

# Convert X into a numpy array if it's not already
X = np.array(X)  # shape (N, 3)
X = X[:, [2, 1, 0]]
y = np.array(y).squeeze()  # shape (N,)

# 2. Split into skin vs non-skin
skin_pixels = X[y == 1]
non_skin_pixels = X[y == 2]

indices = np.random.randint(non_skin_pixels.shape[0], size=skin_pixels.shape[0])
non_skin_pixels = non_skin_pixels[indices, :]

# 3. Convert each set into (r,g) space
skin_rg = to_rg_space(skin_pixels)
non_skin_rg = to_rg_space(non_skin_pixels)


# 4. Fit two separate GMMs: one for skin, one for non-skin
#    You can tune n_components if desired
n_components = 1
skin_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
non_skin_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)

skin_gmm.fit(skin_rg)
non_skin_gmm.fit(non_skin_rg)

# Optionally, save the models
joblib.dump(skin_gmm, 'skin_gmm.joblib')
joblib.dump(non_skin_gmm, 'non_skin_gmm.joblib')

print("Trained skin GMM and non-skin GMM successfully.")

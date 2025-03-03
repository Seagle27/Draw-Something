import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


def to_rg_space(rgb_array):
    R = rgb_array[:, 0].astype(float)
    G = rgb_array[:, 1].astype(float)
    B = rgb_array[:, 2].astype(float)
    denom = R + G + B + 1e-6
    r = R / denom
    g = G / denom
    return np.column_stack((r, g))


# -------------- Data fetching and conversion --------------
skin_segmentation = fetch_ucirepo(id=229)

X = np.array(skin_segmentation.data.features)  # shape (N, 3)
X = X[:, [2, 1, 0]]
y = np.array(skin_segmentation.data.targets).squeeze()  # 1=skin, 2=non-skin

# -------------- Make subsets of data --------------
skin_pixels = X[y == 1]
non_skin_pixels = X[y == 2]

# Sample e.g. 10,000 from each (if you have memory constraints or want faster plots)
rng = np.random.default_rng(42)
skin_subset = skin_pixels[rng.choice(len(skin_pixels), size=10000, replace=False)]
non_skin_subset = non_skin_pixels[rng.choice(len(non_skin_pixels), size=10000, replace=False)]

# Convert each to (r,g)
skin_rg = to_rg_space(skin_subset)
non_skin_rg = to_rg_space(non_skin_subset)

# -------------- Visualization --------------
plt.figure(figsize=(7, 7))
plt.scatter(skin_rg[:, 0], skin_rg[:, 1], s=3, color='red', alpha=0.4, label='Skin')
plt.scatter(non_skin_rg[:, 0], non_skin_rg[:, 1], s=3, color='blue', alpha=0.4, label='Non-Skin')
plt.title("Skin vs. Non-Skin in (r,g) Chromaticity Space")
plt.xlabel("r")
plt.ylabel("g")
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.legend(loc='best')
plt.show()

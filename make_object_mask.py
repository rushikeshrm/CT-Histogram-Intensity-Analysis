import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def main():
    ap = argparse.ArgumentParser(description="Create a mask from a CT silce.")
    ap.add_argument("folder", help="folder with the .tif slices")
    ap.add_argument("--slice", type=int, default=None,
                    help="slice index (default: middle slice)")
    ap.add_argument("--threshold", type=float, default=50,
                    help="keep pixels above this value (default 50)")
    ap.add_argument("--out", default="masks/braze_object_mask.tif",
                    help="output mask file")
    a = ap.parse_args()

    folder = os.path.expanduser(a.folder)
    files = sorted(glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.tiff")))
    if not files:
        sys.exit(f"No .tif files found in {folder}")
    idx = len(files) // 2 if a.slice is None else a.slice
    if not 0 <= idx < len(files):
        sys.exit(
            f"Slice {idx} does not exist; the folder has {len(files)} slices (0 to {len(files) - 1}).")

    img = np.array(Image.open(files[idx]))
    if img.ndim == 3:          # Use one channel for grayscale data.
        img = img[..., 0]

    mask = img > a.threshold
    if not mask.any():
        sys.exit(
            f"No pixels above {a.threshold} in slice {idx}. Try a lower --threshold.")
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    labels, n = ndi.label(mask)
    if n > 1:                  # keep the largest connected region.
        sizes = ndi.sum(mask, labels, range(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    mask = ndi.binary_fill_holes(mask)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(a.out)
    print(f"Using slice: {idx} ({os.path.basename(files[idx])})")
    print(f"Image size: {img.shape[1]} x {img.shape[0]}")
    print(
        f"Pixels in mask: {int(mask.sum())} ({100 * mask.mean():.1f}% of the slice)")
    print(f"Mask save to: {a.out}")


if __name__ == "__main__":
    main()

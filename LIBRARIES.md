# Libraries Used

## 1. OpenCV 5.0.0 (C++)
- Used for: reading the `.tif` CT slices
- Install (macOS): `brew install opencv pkg-config`
- Website: https://opencv.org

## 2. C++17 standard library
- No installation needed
- Used for: file handling, timing, vectors, math

## 3. Python 3 packages (only for `plot_histograms.py`)
- pandas
- matplotlib
- Install: `pip install -r requirements.txt`

## 4. CUDA Toolkit (optional, GPU path only)
- Needed only if the program is built with `-DUSE_CUDA`
- The GPU path has not been tested

## Tested on
macOS (Apple Silicon), Homebrew OpenCV 5.0.0

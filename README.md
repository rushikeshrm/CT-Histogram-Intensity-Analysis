# CT Histogram and Intensity Analysis

A C++ command-line tool that reads a CT volume stored as a folder of 2-D `.tif` slices and computes an intensity histogram and statistics. It can analyse the whole volume, a single slice, a range of slices, one row of statistics per slice, a rectangular or circular region of interest (ROI), a mask image, or only the voxels inside an intensity range. The slices are processed one at a time, so volumes with billions of voxels fit in ordinary memory.

Two small Python scripts support it: `plot_histograms.py` plots the results, and `make_object_mask.py` builds a mask that follows the object in a slice.

## Files

| File | Purpose |
|---|---|
| `ct_histogram.cpp` | Main program (C++17) |
| `plot_histograms.py` | Plots histograms, per-slice statistics and dataset comparisons |
| `make_object_mask.py` | Builds a black-and-white object mask from one slice |
| `LIBRARIES.md` | Libraries used and how to install them |
| `requirements.txt` | Python packages |
| `masks/` | Mask images used in the examples |
| `sample_output/` | Example results from real runs (CSV tables and plots) |

`sample_output/` contains `whole_volume/` (one CSV per dataset), `per_slice/` (per-slice CSVs), `roi_mask_examples/` (slice, ROI, circle, mask and intensity-range runs on the Braze dataset) and `plots/`.

## Requirements

- C++17 compiler (`g++` or `clang++`)
- OpenCV and `pkg-config` (tested with OpenCV 5.0.0 on macOS)
- Python 3 with the packages in `requirements.txt` (only for plotting and mask making)

macOS: `brew install opencv pkg-config`
Ubuntu/Debian: `sudo apt install libopencv-dev pkg-config`

Python packages: `pip install -r requirements.txt`

See `LIBRARIES.md` for details.

## Build

```
g++ -O2 -std=c++17 ct_histogram.cpp -o ct_histogram `pkg-config --cflags --libs opencv5`
```

Use `opencv4` instead of `opencv5` if that is the installed version. If `pkg-config` cannot find OpenCV on macOS (Apple Silicon), run this first:

```
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/opt/opencv/lib/pkgconfig:$PKG_CONFIG_PATH"
```

## Usage

```
./ct_histogram <slice_folder> [options]
```

| Option | Meaning |
|---|---|
| `--bins N` | Number of histogram bins (default 256) |
| `--csv out.csv` | Write the histogram and statistics to a CSV file |
| `--per-slice out.csv` | Write one statistics row per slice |
| `--slice N` | Analyse only slice N |
| `--slices A-B` | Analyse slices A to B (inclusive) |
| `--roi x,y,w,h` | Rectangular region of interest |
| `--circle cx,cy,r` | Circular region of interest |
| `--mask path` | Mask image file (used for every slice) or a folder of masks (one per slice) |
| `--min-intensity V` | Ignore voxels below V |
| `--max-intensity V` | Ignore voxels above V |
| `--gpu` | Use the GPU path (only if built with CUDA) |
| `--quiet` | Less progress output |

Notes:

- Slice numbers and pixel coordinates are 0-based. `x` is the column and `y` is the row.
- Slice order is the alphabetical order of the file names (for example `Full000.tif`, `Full001.tif`, ...).
- Use only one of `--roi`, `--circle` and `--mask`. Any of them can be combined with slice selection and with the intensity range.
- In a mask image, any non-zero pixel counts as inside the region. The mask must have the same width and height as the slices. A mask folder must contain exactly as many images as the slice folder, paired in sorted order.
- Slices with 3 identical channels are converted to grayscale.

### Examples

```
# whole volume
./ct_histogram Assembly --csv Assembly_stats.csv --per-slice Assembly_slices.csv

# one slice
./ct_histogram Braze --slice 322 --csv Braze_slice322_stats.csv

# rectangle on a range of slices
./ct_histogram Braze --slices 297-347 --roi 688,848,1376,1697 --csv Braze_roi_stats.csv

# circle on all slices
./ct_histogram Braze --circle 1376,1697,917 --csv Braze_circle_stats.csv

# ignore the black background (keep intensities from 1 up)
./ct_histogram Braze --min-intensity 1 --csv Braze_intensity_stats.csv

# mask image
./ct_histogram Braze --mask masks/braze_mask.tif --csv Braze_mask_stats.csv

# object mask and intensity range together
./ct_histogram Braze --mask masks/braze_object_mask.tif --min-intensity 1 --csv Braze_object_mask_range_stats.csv
```

## Output

**Console:** voxel count, min, max, mean, variance, standard deviation, skewness, excess kurtosis, median, 5th and 95th percentile, and entropy. For 12 slices or fewer, a per-slice table is printed as well.

**`--csv`:** the statistics above, comment lines recording the region and slices used, and the histogram as `bin_center,count`.

**`--per-slice`:** one row per slice with `slice_index, file, voxels, min, max, mean, stddev`.

The median and percentiles are estimated from the histogram, so their accuracy depends on the bin width.

## Plotting

```
python plot_histograms.py Metal_stats.csv Ceramic_stats.csv Braze_slices.csv
```

- A histogram CSV gives `<name>_histogram.png` (log-scaled counts).
- A per-slice CSV gives a plot of mean and standard deviation against slice index.
- Two or more histogram CSVs also give `comparison_histogram.png`, with each intensity axis rescaled to 0-1, so datasets with different bit depths can be compared by shape.
- With no arguments, the script plots every `*_stats.csv` and `*_slices.csv` in the current folder.

## Making an object mask

```
python make_object_mask.py Braze --out masks/braze_object_mask.tif
```

The script takes the middle slice (or `--slice N`), keeps pixels above a threshold (`--threshold`, default 50), closes small gaps, keeps the largest blob and fills holes. The mask comes from one slice and is used on all slices, so it only fits the object where the object stays in place.

## How it works

The volume is processed one slice at a time.

1. **Pass 1** finds the minimum, maximum and mean of the selected voxels. This sets the histogram range and a shift value.
2. **Pass 2** builds the histogram and the sums used for variance, skewness and kurtosis, computed around the shift value for numerical stability, and records per-slice statistics.

Each slice is read as raw values (`IMREAD_UNCHANGED`, no rescaling). 8-bit, 16-bit and floating-point images are supported.

## CPU and GPU

The CPU path is the default and always works. The GPU (CUDA) path is optional and is compiled in only when built with `nvcc -DUSE_CUDA`. It handles whole-slice histograms of 16-bit images; ROI, mask and intensity-range analysis always run on the CPU. **The GPU path has not been tested**, because it needs an NVIDIA GPU. Without CUDA, `--gpu` is ignored.

```
nvcc -O2 -std=c++17 -x cu -DUSE_CUDA ct_histogram.cpp -o ct_histogram `pkg-config --cflags --libs opencv5`
```

## Sample results

Whole-volume statistics (Reconstruction is 16-bit, the others are 8-bit):

| Dataset | Mean | Std dev | Skewness | Excess kurtosis |
|---|---|---|---|---|
| Reconstruction | 10,819 | 5,153 | 2.20 | 3.59 |
| Assembly | 48.3 | 22.1 | 0.44 | 8.85 |
| Braze | 0.60 | 10.6 | 18.9 | 376 |
| Ceramic | 14.6 | 25.3 | 1.61 | 4.72 |
| Metal | 7.46 | 25.1 | 3.41 | 12.6 |
| Outer | 0.39 | 7.90 | 26.4 | 780 |

For Braze and Outer, at least 95 percent of voxels are exactly 0, which explains the extreme skewness and kurtosis. The Braze dataset analysed in different ways:

| Analysed | Mean | Std dev |
|---|---|---|
| Whole volume | 0.60 | 10.6 |
| Slice 322 | 0.7 | 10.9 |
| Rectangle, slices 297-347 | 1.4 | 14.1 |
| Circle, all slices | 1.4 | 14.8 |
| Intensity range 1 to 255 | 164.4 | 59.4 |
| Circular mask | 1.37 | 14.75 |
| Object mask | 1.30 | 13.69 |
| Object mask and intensity range | 142.9 | 20.3 |

Results for every dataset are in `sample_output/`.

## Testing

- Run on real data (macOS, Apple Silicon, OpenCV 5.0.0): the Reconstruction and Assembly volumes gave identical statistics before and after the ROI and mask features were added.
- Statistics were compared with NumPy/SciPy on synthetic 16-bit volumes for the whole volume, a single slice, a slice range, a rectangle, a circle, a mask file, a mask folder, an intensity range, combined options, per-slice output and the histogram counts. 
- Invalid input (a slice that does not exist, an empty region, two region options, a missing mask) stops with an error message.

## Limitations

- Each slice is read twice (once per pass), so large volumes take a while.
- Percentiles and median are histogram-based estimates.
- Slice order depends on file names being sortable (zero-padded numbers).
- The GPU path is untested and covers only whole-slice histograms.
- A rectangle, circle or single mask is the same on every slice, so it may not follow the object everywhere. A folder with one mask per slice is supported, but building such masks is outside this tool.
- Intensities are relative values, not calibrated Hounsfield units, so 8-bit and 16-bit datasets are compared by shape.
- Not validated against the IBSI reference values.


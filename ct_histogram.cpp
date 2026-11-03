#include <opencv2/opencv.hpp>
#if defined(__has_include)
#if __has_include(<opencv2/core/utils/logger.hpp>)
#include <opencv2/core/utils/logger.hpp>
#define HAVE_CV_LOGGER 1
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;
namespace fs = std::filesystem;

struct Histogram
{
    vector<long long> counts;
    double rangeMin = 0.0, rangeMax = 0.0, binWidth = 1.0;
};

// Decide which part of slice to use.
struct Region
{
    enum Type
    {
        FULL,
        RECT,
        CIRCLE,
        MASK
    } type = FULL;
    int x = 0, y = 0, w = 0, h = 0; // RECT
    int cx = 0, cy = 0, r = 0;      // CIRCLE
    string maskPath;                // MASK
    bool useRange = false;          // intensity mask
    double lo = -numeric_limits<double>::max();
    double hi = numeric_limits<double>::max();

    bool isFull() const { return type == FULL && !useRange; }

    string describe() const
    {
        ostringstream s;
        switch (type)
        {
        case FULL:
            s << "whole slice";
            break;
        case RECT:
            s << "rectangle x=" << x << " y=" << y << " w=" << w << " h=" << h;
            break;
        case CIRCLE:
            s << "circle centre=(" << cx << "," << cy << ") radius=" << r;
            break;
        case MASK:
            s << "mask " << maskPath;
            break;
        }
        if (useRange)
        {
            s << ", intensity range [";
            if (lo > -numeric_limits<double>::max())
                s << lo;
            else
                s << "-inf";
            s << ", ";
            if (hi < numeric_limits<double>::max())
                s << hi;
            else
                s << "+inf";
            s << "]";
        }
        return s.str();
    }
};

// Keep the running sum used to calculate the statistics.
// values are shifted first to avoid precision problems.
struct Moments
{
    long long count = 0;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    double minV = numeric_limits<double>::max();
    double maxV = numeric_limits<double>::lowest();
};

struct SliceStat
{
    size_t index = 0;
    string name;
    long long count = 0;
    double minV = 0, maxV = 0, mean = 0, stddev = 0;
};

struct Stats
{
    long long count = 0;
    double minVal = 0, maxVal = 0;
    double mean = 0, variance = 0, stddev = 0, skewness = 0, kurtosis = 0;
    double median = 0, p5 = 0, p95 = 0, entropy = 0;
};

void mergeMoments(Moments &g, const Moments &m)
{
    g.count += m.count;
    g.s1 += m.s1;
    g.s2 += m.s2;
    g.s3 += m.s3;
    g.s4 += m.s4;
    g.minV = min(g.minV, m.minV);
    g.maxV = max(g.maxV, m.maxV);
}

bool isImageExt(string ext)
{
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".tif" || ext == ".tiff" || ext == ".png" || ext == ".bmp" ||
           ext == ".jpg" || ext == ".jpeg";
}

vector<fs::path> collectImageFiles(const string &folder, bool tifOnly)
{
    vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file())
            continue;
        string name = entry.path().filename().string();
        if (!name.empty() && name[0] == '.')
            continue; // hidden files such as ._Full000.tif
        string ext = entry.path().extension().string();
        string lower = ext;
        transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        bool ok = tifOnly ? (lower == ".tif" || lower == ".tiff") : isImageExt(ext);
        if (ok)
            files.push_back(entry.path());
    }
    sort(files.begin(), files.end());
    return files;
}

// Read a slice without changing its pixel depth. Covert colour images to gray.
cv::Mat loadSlice(const fs::path &file, bool &warnedChannels)
{
    cv::Mat img = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
    if (!img.empty() && img.channels() > 1)
    {
        if (!warnedChannels)
        {
            cerr << "Note: slices have " << img.channels() << " channels (e.g. "
                 << file.filename().string() << "); converting to grayscale.\n";
            warnedChannels = true;
        }
        cv::Mat gray;
        cv::cvtColor(img, gray, img.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
        img = gray;
    }
    return img;
}

template <typename T>
void toBinary(const cv::Mat &src, cv::Mat &dst)
{
    for (int r = 0; r < src.rows; ++r)
    {
        const T *s = src.ptr<T>(r);
        uchar *d = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c)
            d[c] = (s[c] != 0) ? 255 : 0;
    }
}

cv::Mat loadMask(const fs::path &path)
{
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty())
        return img;
    if (img.channels() > 1)
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, img.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
        img = gray;
    }
    cv::Mat bin(img.rows, img.cols, CV_8UC1);
    switch (img.depth())
    {
    case CV_8U:
        toBinary<uchar>(img, bin);
        break;
    case CV_8S:
        toBinary<schar>(img, bin);
        break;
    case CV_16U:
        toBinary<ushort>(img, bin);
        break;
    case CV_16S:
        toBinary<short>(img, bin);
        break;
    case CV_32S:
        toBinary<int>(img, bin);
        break;
    case CV_32F:
        toBinary<float>(img, bin);
        break;
    case CV_64F:
        toBinary<double>(img, bin);
        break;
    default:
        return cv::Mat();
    }
    return bin;
}

struct MaskSource
{
    bool enabled = false;
    bool perSlice = false;  // true: one mask file per slice
    cv::Mat single;         // used when !perSlice
    vector<fs::path> files; // used when perSlice
};

// Find the part of this row that could be inside your selected region.
// The mask and intensity limits are checked later for each pixel.
inline bool rowSpan(int r, int cols, const Region &reg, int &c0, int &c1)
{
    c0 = 0;
    c1 = cols;
    if (reg.type == Region::RECT)
    {
        if (r < reg.y || r >= reg.y + reg.h)
            return false;
        c0 = max(0, reg.x);
        c1 = min(cols, reg.x + reg.w);
    }
    else if (reg.type == Region::CIRCLE)
    {
        const long long dy = r - reg.cy;
        const long long d2 = static_cast<long long>(reg.r) * reg.r - dy * dy;
        if (d2 < 0)
            return false;
        const int half = static_cast<int>(floor(sqrt(static_cast<double>(d2))));
        c0 = max(0, reg.cx - half);
        c1 = min(cols, reg.cx + half + 1);
    }
    return c0 < c1;
}

template <typename T, bool DoHist>
void processSliceCPU(const cv::Mat &img, const Region &reg, const cv::Mat &mask,
                     double shift, Histogram &hist, Moments &out)
{
    const int nbins = static_cast<int>(hist.counts.size());
    const double invBin = 1.0 / hist.binWidth;
    const bool useMask = (reg.type == Region::MASK);
    const bool useRange = reg.useRange;

    long long n = 0;
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    double mn = numeric_limits<double>::max();
    double mx = numeric_limits<double>::lowest();

    for (int r = 0; r < img.rows; ++r)
    {
        int c0, c1;
        if (!rowSpan(r, img.cols, reg, c0, c1))
            continue;
        const T *row = img.ptr<T>(r);
        const uchar *mrow = useMask ? mask.ptr<uchar>(r) : nullptr;
        for (int c = c0; c < c1; ++c)
        {
            if (useMask && mrow[c] == 0)
                continue;
            const double v = static_cast<double>(row[c]);
            if (useRange && (v < reg.lo || v > reg.hi))
                continue;

            const double d = v - shift;
            const double d2 = d * d;
            ++n;
            s1 += d;
            s2 += d2;
            s3 += d2 * d;
            s4 += d2 * d2;
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
            if (DoHist)
            {
                int bin = static_cast<int>((v - hist.rangeMin) * invBin);
                if (bin < 0)
                    bin = 0;
                if (bin >= nbins)
                    bin = nbins - 1;
                hist.counts[bin]++;
            }
        }
    }
    out.count = n;
    out.s1 = s1;
    out.s2 = s2;
    out.s3 = s3;
    out.s4 = s4;
    out.minV = mn;
    out.maxV = mx;
}

template <bool DoHist>
bool dispatchSlice(const cv::Mat &img, const Region &reg, const cv::Mat &mask,
                   double shift, Histogram &hist, Moments &out)
{
    switch (img.depth())
    {
    case CV_8U:
        processSliceCPU<uchar, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    case CV_8S:
        processSliceCPU<schar, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    case CV_16U:
        processSliceCPU<ushort, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    case CV_16S:
        processSliceCPU<short, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    case CV_32S:
        processSliceCPU<int, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    case CV_32F:
        processSliceCPU<float, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    case CV_64F:
        processSliceCPU<double, DoHist>(img, reg, mask, shift, hist, out);
        return true;
    default:
        cerr << "Unsupported pixel depth: " << img.depth() << "\n";
        return false;
    }
}

// OPTIONAL GPU PATH (compiled only with -DUSE_CUDA via nvcc; NOT tested)
// Whole-slice histogram on the GPU (one thread per pixel, atomicAdd into bins).
// Moments are still summed on the CPU. 16-bit slices only; anything else
// falls back to the CPU path so results stay correct.

#ifdef USE_CUDA

__global__ void histogramKernel(const unsigned short *data, int numPixels,
                                unsigned long long *bins, int nbins,
                                double rangeMin, double invBin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels)
        return;
    int bin = static_cast<int>((static_cast<double>(data[idx]) - rangeMin) * invBin);
    if (bin < 0)
        bin = 0;
    if (bin >= nbins)
        bin = nbins - 1;
    atomicAdd(&bins[bin], 1ULL);
}

void processSliceGPU(const cv::Mat &img, const Region &reg, const cv::Mat &mask,
                     double shift, Histogram &hist, Moments &out)
{
    if (img.depth() != CV_16U || !img.isContinuous())
    {
        dispatchSlice<true>(img, reg, mask, shift, hist, out);
        return;
    }
    const int numPixels = img.rows * img.cols;
    const int nbins = static_cast<int>(hist.counts.size());

    unsigned short *d_data = nullptr;
    unsigned long long *d_bins = nullptr;
    cudaMalloc(&d_data, numPixels * sizeof(unsigned short));
    cudaMalloc(&d_bins, nbins * sizeof(unsigned long long));
    cudaMemcpy(d_data, img.ptr<unsigned short>(0),
               numPixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, nbins * sizeof(unsigned long long));

    const int threads = 256;
    const int blocks = (numPixels + threads - 1) / threads;
    histogramKernel<<<blocks, threads>>>(d_data, numPixels, d_bins, nbins,
                                         hist.rangeMin, 1.0 / hist.binWidth);
    cudaDeviceSynchronize();

    vector<unsigned long long> sliceCounts(nbins);
    cudaMemcpy(sliceCounts.data(), d_bins, nbins * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < nbins; ++i)
        hist.counts[i] += static_cast<long long>(sliceCounts[i]);
    cudaFree(d_data);
    cudaFree(d_bins);

    Histogram unused; // moments only (histogram already done on the GPU)
    unused.counts.assign(1, 0);
    dispatchSlice<false>(img, reg, mask, shift, unused, out);
}

#endif // USE_CUDA

// Calculate the final statistics.

// Estimate a percentile from the histogram. The result depends on the bin width.
double percentileFromHistogram(const Histogram &h, long long total, double fraction)
{
    const double target = fraction * static_cast<double>(total);
    double cumulative = 0.0;
    for (size_t i = 0; i < h.counts.size(); ++i)
    {
        const double nextCum = cumulative + static_cast<double>(h.counts[i]);
        if (nextCum >= target && h.counts[i] > 0)
        {
            const double within = (target - cumulative) / static_cast<double>(h.counts[i]);
            return h.rangeMin + (static_cast<double>(i) + within) * h.binWidth;
        }
        cumulative = nextCum;
    }
    return h.rangeMax;
}

Stats finalizeStats(const Moments &m, double shift, const Histogram &hist)
{
    Stats s;
    s.count = m.count;
    s.minVal = m.minV;
    s.maxVal = m.maxV;
    if (m.count == 0)
        return s;

    const double n = static_cast<double>(m.count);
    const double a1 = m.s1 / n, a2 = m.s2 / n, a3 = m.s3 / n, a4 = m.s4 / n;

    s.mean = shift + a1;
    s.variance = a2 - a1 * a1;
    if (s.variance < 0.0)
        s.variance = 0.0;
    s.stddev = sqrt(s.variance);

    if (s.stddev > 0.0)
    {
        const double m3 = a3 - 3.0 * a1 * a2 + 2.0 * a1 * a1 * a1;
        const double m4 = a4 - 4.0 * a1 * a3 + 6.0 * a1 * a1 * a2 - 3.0 * a1 * a1 * a1 * a1;
        const double var2 = s.variance;
        s.skewness = m3 / (var2 * s.stddev);
        s.kurtosis = m4 / (var2 * var2) - 3.0; // excess kurtosis
    }

    s.median = percentileFromHistogram(hist, m.count, 0.50);
    s.p5 = percentileFromHistogram(hist, m.count, 0.05);
    s.p95 = percentileFromHistogram(hist, m.count, 0.95);

    double entropy = 0.0;
    for (long long c : hist.counts)
    {
        if (c > 0)
        {
            const double p = static_cast<double>(c) / n;
            entropy -= p * log2(p);
        }
    }
    s.entropy = entropy;
    return s;
}

SliceStat makeSliceStat(size_t index, const string &name, const Moments &m, double shift)
{
    SliceStat s;
    s.index = index;
    s.name = name;
    s.count = m.count;
    if (m.count > 0)
    {
        const double n = static_cast<double>(m.count);
        const double a1 = m.s1 / n, a2 = m.s2 / n;
        double var = a2 - a1 * a1;
        if (var < 0.0)
            var = 0.0;
        s.mean = shift + a1;
        s.stddev = sqrt(var);
        s.minV = m.minV;
        s.maxV = m.maxV;
    }
    return s;
}

// Print and save the result.

void printStats(const Stats &s)
{
    cout << setprecision(10);
    cout << "\nStatistics\n";
    cout << "Voxel count : " << s.count << "\n";
    cout << "Min         : " << s.minVal << "\n";
    cout << "Max         : " << s.maxVal << "\n";
    cout << "Mean        : " << s.mean << "\n";
    cout << "Variance    : " << s.variance << "\n";
    cout << "Std Dev     : " << s.stddev << "\n";
    cout << "Skewness    : " << s.skewness << "\n";
    cout << "Kurtosis    : " << s.kurtosis << " (excess)\n";
    cout << "Median      : " << s.median << " (from histogram)\n";
    cout << "5th pct     : " << s.p5 << " (from histogram)\n";
    cout << "95th pct    : " << s.p95 << " (from histogram)\n";
    cout << "Entropy     : " << s.entropy << " bits\n";
}

void printSliceTable(const vector<SliceStat> &v)
{
    cout << "\nPer-slice statistics\n";
    cout << left << setw(8) << "slice" << setw(14) << "voxels" << setw(12) << "min"
         << setw(12) << "max" << setw(14) << "mean" << "std dev\n";
    for (const auto &s : v)
    {
        cout << left << setw(8) << s.index << setw(14) << s.count;
        if (s.count == 0)
        {
            cout << "(no voxels inside the region)\n";
        }
        else
        {
            cout << setw(12) << s.minV << setw(12) << s.maxV
                 << setw(14) << s.mean << s.stddev << "\n";
        }
    }
}

void writeCSV(const string &path, const Histogram &hist, const Stats &s,
              const string &regionText, const string &sliceText)
{
    ofstream out(path);
    if (!out)
    {
        cerr << "Could not open " << path << " for writing.\n";
        return;
    }
    out << setprecision(12);
    out << "stat,value\n";
    out << "count," << s.count << "\n";
    out << "min," << s.minVal << "\n";
    out << "max," << s.maxVal << "\n";
    out << "mean," << s.mean << "\n";
    out << "variance," << s.variance << "\n";
    out << "stddev," << s.stddev << "\n";
    out << "skewness," << s.skewness << "\n";
    out << "kurtosis_excess," << s.kurtosis << "\n";
    out << "median," << s.median << "\n";
    out << "p5," << s.p5 << "\n";
    out << "p95," << s.p95 << "\n";
    out << "entropy_bits," << s.entropy << "\n";
    out << "# region: " << regionText << "\n";
    out << "# slices: " << sliceText << "\n";
    out << "\nbin_center,count\n";
    for (size_t i = 0; i < hist.counts.size(); ++i)
    {
        out << hist.rangeMin + (static_cast<double>(i) + 0.5) * hist.binWidth
            << "," << hist.counts[i] << "\n";
    }
    cout << "Wrote histogram and statistics to " << path << "\n";
}

void writeSliceCSV(const string &path, const vector<SliceStat> &v)
{
    ofstream out(path);
    if (!out)
    {
        cerr << "Could not open " << path << " for writing.\n";
        return;
    }
    out << setprecision(12);
    out << "slice_index,file,voxels,min,max,mean,stddev\n";
    for (const auto &s : v)
    {
        out << s.index << "," << s.name << "," << s.count << ",";
        if (s.count == 0)
            out << ",,,\n";
        else
            out << s.minV << "," << s.maxV << "," << s.mean << "," << s.stddev << "\n";
    }
    cout << "Saved per-slice statistics to " << path << "\n";
}

void printUsage(const char *prog)
{
    cerr << "Usage: " << prog << " <slice_folder> [options]\n"
         << "  --bins N             histogram bins (default 256)\n"
         << "  --csv out.csv        write histogram + statistics\n"
         << "  --per-slice out.csv  write one statistics row per slice\n"
         << "  --slice N            analyse only slice N (0-based)\n"
         << "  --slices A-B         analyse slices A to B (inclusive)\n"
         << "  --roi x,y,w,h        rectangular region of interest\n"
         << "  --circle cx,cy,r     circular region of interest\n"
         << "  --mask path          mask image, or folder of mask images\n"
         << "  --min-intensity V    ignore voxels below V\n"
         << "  --max-intensity V    ignore voxels above V\n"
         << "  --gpu                use GPU path (only if built with CUDA)\n"
         << "  --quiet              less progress output\n";
}

// Program entry point.
int main(int argc, char **argv)
{
    // Silence harmless libtiff metadata warnings (e.g. "Unknown field with tag 33451").
    setenv("OPENCV_LOG_LEVEL", "ERROR", 1);
#ifdef HAVE_CV_LOGGER
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
#endif

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    const string folder = argv[1];
    int nbins = 256;
    bool useGPU = false, quiet = false;
    string csvPath, sliceCsvPath;
    Region region;
    int regionKinds = 0;
    long sliceFrom = -1, sliceTo = -1; // -1 = not set

    for (int i = 2; i < argc; ++i)
    {
        const string arg = argv[i];
        const bool hasNext = (i + 1 < argc);
        if (arg == "--bins" && hasNext)
        {
            nbins = atoi(argv[++i]);
        }
        else if (arg == "--csv" && hasNext)
        {
            csvPath = argv[++i];
        }
        else if (arg == "--per-slice" && hasNext)
        {
            sliceCsvPath = argv[++i];
        }
        else if (arg == "--slice" && hasNext)
        {
            sliceFrom = sliceTo = atol(argv[++i]);
        }
        else if (arg == "--slices" && hasNext)
        {
            long a, b;
            if (sscanf(argv[++i], "%ld-%ld", &a, &b) != 2 || a < 0 || b < a)
            {
                cerr << "--slices expects A-B with 0 <= A <= B, e.g. --slices 100-200\n";
                return 1;
            }
            sliceFrom = a;
            sliceTo = b;
        }
        else if (arg == "--roi" && hasNext)
        {
            int x, y, w, h;
            if (sscanf(argv[++i], "%d,%d,%d,%d", &x, &y, &w, &h) != 4 || w <= 0 || h <= 0)
            {
                cerr << "--roi expects x,y,w,h with w,h > 0, e.g. --roi 100,100,400,300\n";
                return 1;
            }
            region.type = Region::RECT;
            region.x = x;
            region.y = y;
            region.w = w;
            region.h = h;
            ++regionKinds;
        }
        else if (arg == "--circle" && hasNext)
        {
            int cx, cy, r;
            if (sscanf(argv[++i], "%d,%d,%d", &cx, &cy, &r) != 3 || r <= 0)
            {
                cerr << "--circle expects cx,cy,r with r > 0, e.g. --circle 500,500,300\n";
                return 1;
            }
            region.type = Region::CIRCLE;
            region.cx = cx;
            region.cy = cy;
            region.r = r;
            ++regionKinds;
        }
        else if (arg == "--mask" && hasNext)
        {
            region.type = Region::MASK;
            region.maskPath = argv[++i];
            ++regionKinds;
        }
        else if (arg == "--min-intensity" && hasNext)
        {
            region.useRange = true;
            region.lo = atof(argv[++i]);
        }
        else if (arg == "--max-intensity" && hasNext)
        {
            region.useRange = true;
            region.hi = atof(argv[++i]);
        }
        else if (arg == "--gpu")
        {
            useGPU = true;
        }
        else if (arg == "--quiet")
        {
            quiet = true;
        }
        else
        {
            cerr << "Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (nbins < 1)
    {
        cerr << "--bins must be at least 1.\n";
        return 1;
    }
    if (regionKinds > 1)
    {
        cerr << "Use only one of --roi, --circle and --mask.\n";
        return 1;
    }
    if (region.useRange && region.lo > region.hi)
    {
        cerr << "--min-intensity must not be larger than --max-intensity.\n";
        return 1;
    }

    if (!fs::is_directory(folder))
    {
        cerr << "Folder not found: " << folder << "\n";
        return 1;
    }
    const auto allFiles = collectImageFiles(folder, true);
    if (allFiles.empty())
    {
        cerr << "No .tif/.tiff files found in " << folder << "\n";
        return 1;
    }

    // Pick the slices requested by the user.
    size_t first = 0, last = allFiles.size() - 1;
    if (sliceFrom >= 0)
    {
        if (static_cast<size_t>(sliceFrom) >= allFiles.size())
        {
            cerr << "Slice " << sliceFrom << " does not exist; the folder has "
                 << allFiles.size() << " slices (0 to " << allFiles.size() - 1 << ").\n";
            return 1;
        }
        first = static_cast<size_t>(sliceFrom);
        last = min(static_cast<size_t>(sliceTo), allFiles.size() - 1);
    }
    vector<size_t> selected;
    for (size_t i = first; i <= last; ++i)
        selected.push_back(i);

    // Set up the mask, if one was supplied.
    MaskSource masks;
    if (region.type == Region::MASK)
    {
        masks.enabled = true;
        if (fs::is_directory(region.maskPath))
        {
            masks.perSlice = true;
            masks.files = collectImageFiles(region.maskPath, false);
            if (masks.files.size() != allFiles.size())
            {
                cerr << "The mask folder has " << masks.files.size() << " images but the slice "
                     << "folder has " << allFiles.size() << ". They must match one-to-one.\n";
                return 1;
            }
        }
        else if (fs::is_regular_file(region.maskPath))
        {
            masks.single = loadMask(region.maskPath);
            if (masks.single.empty())
            {
                cerr << "Could not read mask image " << region.maskPath << "\n";
                return 1;
            }
        }
        else
        {
            cerr << "Mask path not found: " << region.maskPath << "\n";
            return 1;
        }
    }

    ostringstream sliceTextS;
    if (selected.size() == allFiles.size())
        sliceTextS << "all " << allFiles.size() << " slices";
    else
        sliceTextS << selected.size() << " slice(s), index " << first << " to " << last
                   << " of " << allFiles.size();
    const string sliceText = sliceTextS.str();

    cout << "Folder  : " << folder << "\n";
    cout << "Slices  : " << sliceText << "\n";
    cout << "Region  : " << region.describe() << "\n";

    // Get the mask for this slice, if masking is enabled.
    auto maskFor = [&](size_t fileIndex, cv::Mat &out, const cv::Mat &img) -> bool
    {
        if (!masks.enabled)
            return true;
        if (masks.perSlice)
        {
            out = loadMask(masks.files[fileIndex]);
            if (out.empty())
            {
                cerr << "Could not read mask " << masks.files[fileIndex].filename().string() << "\n";
                return false;
            }
        }
        else
        {
            out = masks.single;
        }
        if (out.rows != img.rows || out.cols != img.cols)
        {
            cerr << "Mask size (" << out.cols << "x" << out.rows << ") does not match slice "
                 << allFiles[fileIndex].filename().string() << " (" << img.cols << "x"
                 << img.rows << ").\n";
            return false;
        }
        return true;
    };

    bool warnedChannels = false;
    const auto t0 = chrono::high_resolution_clock::now();

    // First pass: find the intensity range and average.
    cout << "Pass 1/2: scanning intensity range...\n";
    double minVal = numeric_limits<double>::max();
    double maxVal = numeric_limits<double>::lowest();
    double totalSum = 0.0;
    long long totalCount = 0;
    size_t readOK = 0;
    cv::Mat mask;

    for (size_t k = 0; k < selected.size(); ++k)
    {
        const size_t idx = selected[k];
        cv::Mat img = loadSlice(allFiles[idx], warnedChannels);
        if (img.empty())
        {
            cerr << "Warning: could not read " << allFiles[idx].filename().string() << ", skipping.\n";
            continue;
        }
        ++readOK;
        if (region.isFull())
        {
            double mn, mx;
            cv::minMaxLoc(img, &mn, &mx);
            minVal = min(minVal, mn);
            maxVal = max(maxVal, mx);
            totalSum += cv::sum(img)[0];
            totalCount += static_cast<long long>(img.rows) * img.cols;
        }
        else
        {
            if (!maskFor(idx, mask, img))
                return 1;
            Moments m;
            Histogram dummy;
            dummy.counts.assign(1, 0);
            if (!dispatchSlice<false>(img, region, mask, 0.0, dummy, m))
                return 1;
            if (m.count > 0)
            {
                minVal = min(minVal, m.minV);
                maxVal = max(maxVal, m.maxV);
                totalSum += m.s1; // shift is 0, so s1 is the plain sum
                totalCount += m.count;
            }
        }
        if (!quiet && selected.size() > 1 && ((k + 1) % 100 == 0 || k + 1 == selected.size()))
            cout << "  " << (k + 1) << "/" << selected.size() << " slices\n";
    }
    if (readOK == 0)
    {
        cerr << "No readable slices.\n";
        return 1;
    }
    if (totalCount == 0)
    {
        cerr << "No voxels fall inside the selected region (check the ROI, mask and "
                "intensity range).\n";
        return 1;
    }
    const double shift = totalSum / static_cast<double>(totalCount);

    Histogram hist;
    hist.counts.assign(nbins, 0);
    hist.rangeMin = minVal;
    hist.rangeMax = (maxVal > minVal) ? maxVal : minVal + 1.0;
    hist.binWidth = (hist.rangeMax - hist.rangeMin) / nbins;

    // Second pass: fill the histogram and calculate the statistics.
    const bool gpuOK = useGPU && region.isFull();
    if (useGPU && !gpuOK)
        cerr << "Note: the GPU path only handles whole-slice histograms; using the CPU for this region.\n";
    cout << "Pass 2/2: histogram and statistics (" << (gpuOK ? "GPU" : "CPU")
         << " path, " << nbins << " bins)...\n";

    Moments total;
    vector<SliceStat> sliceStats;
    for (size_t k = 0; k < selected.size(); ++k)
    {
        const size_t idx = selected[k];
        cv::Mat img = loadSlice(allFiles[idx], warnedChannels);
        if (img.empty())
            continue;
        if (!maskFor(idx, mask, img))
            return 1;

        Moments m;
#ifdef USE_CUDA
        if (gpuOK)
            processSliceGPU(img, region, mask, shift, hist, m);
        else
            dispatchSlice<true>(img, region, mask, shift, hist, m);
#else
        dispatchSlice<true>(img, region, mask, shift, hist, m);
#endif
        mergeMoments(total, m);
        sliceStats.push_back(makeSliceStat(idx, allFiles[idx].filename().string(), m, shift));

        if (!quiet && selected.size() > 1 && ((k + 1) % 100 == 0 || k + 1 == selected.size()))
            cout << "  " << (k + 1) << "/" << selected.size() << " slices\n";
    }

    const double seconds =
        chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();
    cout << "Slices used: " << readOK << " of " << selected.size()
         << "   Total time: " << fixed << setprecision(2) << seconds << " s\n"
         << defaultfloat;

    const Stats stats = finalizeStats(total, shift, hist);
    printStats(stats);
    if (sliceStats.size() <= 12)
        printSliceTable(sliceStats);

    if (!csvPath.empty())
        writeCSV(csvPath, hist, stats, region.describe(), sliceText);
    if (!sliceCsvPath.empty())
        writeSliceCSV(sliceCsvPath, sliceStats);
    return 0;
}

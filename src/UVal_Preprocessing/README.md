# MicrobAI - Mass Spectrometry Bacterial Classification

A Python pipeline for processing mass spectrometry data to classify bacterial samples using machine learning techniques. This project processes MS2 data from bacterial cultures and generates feature matrices for downstream classification tasks.

## Author
Simon Pelletier

## Overview

MicrobAI is designed to process mass spectrometry TSV files containing bacterial MS2 data and convert them into structured tensor representations suitable for machine learning. The pipeline includes data binning, feature selection, preprocessing, and sparse matrix operations for efficient handling of large-scale MS data.

## Features

- **Mass Spectrometry Data Processing**: Converts TSV files with RT (retention time), m/z (mass-to-charge ratio), and intensity data into structured tensors
- **Flexible Binning**: Configurable binning strategies for RT and m/z dimensions
- **Feature Selection**: Multiple feature selection methods including mutual information and f-statistics
- **Sparse Matrix Support**: Efficient handling of sparse matrices for memory optimization
- **Data Preprocessing**: Various scaling and normalization options
- **Batch Processing**: Multi-core processing support for large datasets
- **Peak Processing**: Optional peak detection and alignment

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- pillow>=8.0.0
- tqdm>=4.62.0
- msalign (optional, for peak alignment)

## Project Structure

```
microbAI/
├── make_tensors_sparse_ms2.py          # Main processing script
├── features_selection.py               # Feature selection utilities
├── features_selection_sparse.py        # Sparse matrix feature selection
├── utils.py                           # Utility functions
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── resources/                        # Data directory
    ├── B1-02-02-2024/               # Batch data folders
    ├── B2-02-21-2024/
    ├── ...
    ├── culturespures/               # Pure culture data
    └── *.csv                        # Sample metadata files
```

## Usage

### Basic Usage

Run the main processing script:

```bash
python make_tensors_sparse_ms2.py
```

### Command Line Arguments

#### Data Processing Parameters

- `--test_run` (int, default=1): Enable test run mode (1) or full processing (0)
- `--n_samples` (int, default=-1): Number of samples to process (-1 for all)
- `--resources_path` (str, default='resources'): Path to input data directory
- `--experiment` (str, default=''): Experiment name for output organization
- `--make_data` (int, default=1): Whether to generate new data matrices (1) or load existing (0)

#### Binning Parameters

- `--mz_bin_post` (float, default=10): m/z binning size
- `--rt_bin_post` (float, default=320): Retention time binning size
- `--shift` (int, default=0): Apply shifting to bin centers
- `--min_mz` (int, default=0): Minimum m/z value to include
- `--max_mz` (int, default=10000): Maximum m/z value to include
- `--min_rt` (int, default=0): Minimum retention time to include
- `--max_rt` (int, default=320): Maximum retention time to include

#### Intensity Aggregation

- `--aggregation_method` (str, default="logaddinloop"): How to combine intensities
  - `"logaddinloop"`: Apply log1p during binning
  - `"logmax"`: Take log of maximum intensity
  - `"logaddafterloop"`: Apply log1p after binning
  - `"max"`: Take maximum intensity
  - `"no"`: Simple addition

#### Feature Selection

- `--feature_selection` (str, default='none'): Feature selection method
  - `"variance"`: Variance threshold (recommended)
  - `"none"`: No feature selection
- `--feature_selection_threshold` (float, default=0.0): Threshold for feature selection
- `--k` (int, default=-1): Number of top features to keep (-1 for all)

#### Data Preprocessing

- `--scaler` (str, default="none"): Data scaling method
- `"robust"`: Robust scaling (median and IQR)
  - `"standard"`: Z-score standardization  
  - `"minmax"`: Min-max normalization
  - `"none"`: No scaling
- `--threshold` (float, default=0.0): Threshold for removing sparse features
- `--decimals` (int, default=-1): Number of decimal places for rounding (-1 to disable)

#### Signal Processing

- `--find_peaks` (int, default=0): Apply peak detection (1) or not (0)
- `--align_peaks` (int, default=0): Apply peak alignment (1) or not (0)

#### Performance

- `--n_cpus` (int, default=-1): Number of CPU cores to use (-1 for all available)
- `--is_sparse` (int, default=1): Use sparse matrices (1) or dense (0)

#### Output Options

- `--save` (int, default=1): Save output images and CSV files (1) or not (0)
- `--save3d` (int, default=0): Save 3D representations (1) or not (0)
- `--run_name` (str, default="all"): Name for the analysis run

### Example Commands

#### Basic processing with default parameters:
```bash
python make_tensors_sparse_ms2.py --experiment "bacterial_classification"
```

#### Test run with limited samples:
```bash
python make_tensors_sparse_ms2.py --test_run 1 --n_samples 10
```

#### Processing with variance-based feature selection:
```bash
python make_tensors_sparse_ms2.py \
    --test_run 0 \
    --mz_bin_post 5 \
    --rt_bin_post 20 \
    --feature_selection "variance" \
    --feature_selection_threshold 0.1 \
    --k 1000 \
    --scaler "robust" \
    --experiment "optimized_run"
```

#### Processing with peak detection:
```bash
python make_tensors_sparse_ms2.py \
    --find_peaks 1 \
    --aggregation_method "logmax" \
    --scaler "standard"
```

## Input Data Format

The pipeline expects TSV files with the following columns:
- `min_parent_mz`: Minimum parent m/z value
- `rt_bin`: Retention time bin
- `mz_bin`: m/z bin  
- `bin_intensity`: Intensity value for the bin
- `max_parent_mz`: Maximum parent m/z value (optional)

Files should be organized in batch directories under the resources folder.

## Output

The pipeline generates:

- **Data matrices**: Sparse matrices with binned MS data
- **Feature scores**: CSV files with feature selection results  
- **Processed datasets**: CSV files ready for machine learning
- **Images**: PNG visualizations of the MS data (optional)
- **Logs**: Processing logs and metadata

Output files are saved in a structured directory hierarchy:
```
resources/[experiment]/matrices/mzp[mz_bin]/rtp[rt_bin]/thr[threshold]/...
```

## Module Descriptions

- **`make_tensors_sparse_ms2.py`**: Main processing pipeline
- **`features_selection.py`**: Feature selection algorithms for dense matrices
- **`features_selection_sparse.py`**: Feature selection optimized for sparse matrices
- **`utils.py`**: Utility functions for data cropping and tensor adjustment

## Performance Tips

1. Use sparse matrices (`--is_sparse 1`) for memory efficiency
2. Adjust `--n_cpus` based on your system capabilities
3. Use appropriate binning sizes to balance resolution and performance
4. Consider feature selection to reduce dimensionality for large datasets

## Troubleshooting

- **Memory issues**: Reduce bin resolution or enable sparse matrices
- **Long processing times**: Increase bin sizes or reduce number of samples for testing
- **Missing features**: Check that input TSV files have the required columns
- **Peak alignment errors**: Ensure msalign package is properly installed

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Citation

[Add citation information here]

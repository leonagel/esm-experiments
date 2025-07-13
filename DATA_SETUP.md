# Dataset Setup

This directory uses CIFAR-10 and CIFAR-100 datasets. To avoid downloading them multiple times during training, we provide scripts to download them once.

## Quick Setup

```bash
# Download all datasets to ./data directory
./download_datasets.sh

# Or specify a custom data directory
./download_datasets.sh --data-dir /path/to/data

# Check if datasets are already downloaded
./download_datasets.sh --check-only
```

## Python Script Usage

You can also use the Python script directly:

```bash
# Download datasets
python download_datasets.py

# Specify custom directory
python download_datasets.py --data-dir /path/to/data

# Check status only
python download_datasets.py --check-only
```

## Dataset Information

- **CIFAR-10**: 60,000 32x32 color images in 10 classes
  - 50,000 training images
  - 10,000 test images
  - Size: ~170 MB

- **CIFAR-100**: 60,000 32x32 color images in 100 classes
  - 50,000 training images
  - 10,000 test images
  - Size: ~170 MB

Total download size: ~340 MB

## Configuration

When running training scripts, ensure your config files point to the correct data directory:

```yaml
dataset:
  root: ./data  # Should match where you downloaded the data
  download: false  # Set to false if already downloaded
```

## Troubleshooting

If you encounter download errors:
1. Check your internet connection
2. Ensure you have write permissions to the data directory
3. Try downloading again - the script will skip already downloaded files
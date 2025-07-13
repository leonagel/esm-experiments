#!/bin/bash
# Download CIFAR-10 and CIFAR-100 datasets to local data directory

set -e  # Exit on error

echo "======================================"
echo "CIFAR Dataset Download Script"
echo "======================================"

# Default data directory
DATA_DIR="${DATA_DIR:-./data}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --data-dir DIR    Directory to save datasets (default: ./data)"
            echo "  --check-only      Only check if datasets exist"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if torchvision is installed
if ! python3 -c "import torchvision" &> /dev/null; then
    echo "Error: torchvision is not installed"
    echo "Please install with: pip install torchvision"
    exit 1
fi

# Run the download script
if [ "$CHECK_ONLY" = true ]; then
    echo "Checking dataset status..."
    python3 download_datasets.py --data-dir "$DATA_DIR" --check-only
else
    echo "Downloading datasets to: $DATA_DIR"
    echo ""
    python3 download_datasets.py --data-dir "$DATA_DIR"
fi

exit $?
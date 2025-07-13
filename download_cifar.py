#!/usr/bin/env python3
import os
import tarfile
import urllib.request
from pathlib import Path

def download_file(url, filepath):
    """Download a file from URL to filepath with progress bar."""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"Downloading: {percent:.1f}% [{downloaded}/{total_size} bytes]", end='\r')
    
    urllib.request.urlretrieve(url, filepath, reporthook=download_progress)
    print()  # New line after download completes

def extract_tar(filepath, extract_to):
    """Extract tar.gz file."""
    print(f"Extracting {filepath}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def download_cifar(dataset_name, url, data_dir):
    """Download and extract CIFAR dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filename = url.split('/')[-1]
    filepath = data_dir / filename
    
    if filepath.exists():
        print(f"{dataset_name} archive already exists at {filepath}")
    else:
        print(f"Downloading {dataset_name}...")
        download_file(url, filepath)
    
    # Check if already extracted
    extract_dir = data_dir / filename.replace('.tar.gz', '')
    if extract_dir.exists():
        print(f"{dataset_name} already extracted at {extract_dir}")
    else:
        extract_tar(filepath, data_dir)
    
    print(f"{dataset_name} ready at {data_dir}\n")

def main():
    # URLs for CIFAR datasets
    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    
    # Data directory
    data_dir = "./data"
    
    print("Starting CIFAR dataset downloads...\n")
    
    # Download CIFAR-10
    download_cifar("CIFAR-10", cifar10_url, data_dir)
    
    # Download CIFAR-100
    download_cifar("CIFAR-100", cifar100_url, data_dir)
    
    print("All downloads completed!")

if __name__ == "__main__":
    main()
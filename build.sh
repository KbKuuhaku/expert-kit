#!/bin/bash 

# make sure these directories exist

VENDOR_DIR="vendor"
if [[ ! -d $VENDOR_DIR ]]; then
    echo "Vendor directory not exists, creating..."
    mkdir $VENDOR_DIR
fi 

EK_CACHE_DIR="/tmp/expert-kit/cache"

if [[ ! -d $EK_CACHE_DIR ]]; then
    echo "EK cache directory not exists, creating..."
    mkdir -p $EK_CACHE_DIR
fi

# download libtorch from https://pytorch.org/, and place it in the vendor directory of expert-kit
# Mac: https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip
# Linux: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip

LIBTORCH_ZIP="/tmp/libtorch.zip"
if [[ ! -d $VENDOR_DIR/libtorch ]]; then
    echo "LibTorch does not exist, fetching..."
    wget "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip" -O $LIBTORCH_ZIP
    unzip $LIBTORCH_ZIP -d $VENDOR_DIR
fi

# Set environment variables
./set_env.sh

cargo build --release




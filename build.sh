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

if [[ ! -d $VENDOR_DIR/libtorch ]]; then
    echo "LibTorch does not exist, fetching..."
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -O /tmp/libtorch.zip
    unzip /tmp/libtorch.zip -d ./vendor/
fi

export LIBTORCH=$(realpath ./vendor/libtorch)
export LD_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)
export DYLD_FALLBACK_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)
export EK_CONFIG="$(realpath ./dev/hello-world.config.yaml)"

# download the expert-kit source code
# since we are using git lfs, make sure you have git-lfs installed and initialized
git lfs fetch --all  # download the ds-tiny weight
git lfs install      # initialize git-lfs if not done yet
git lfs checkout     # checkout the ds-tiny weight files

cargo build --release
uv sync

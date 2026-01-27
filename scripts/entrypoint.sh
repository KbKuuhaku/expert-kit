#!/bin/bash

# Move into "/etc/environment" all environment variables defined in dockerfile
# /etc/environment is like a global KEY-VALUE pairs which could be defined in .bashrc
echo "Syncing ENV from the Dockerfile..."
echo "source /etc/environment.custom" >> "/etc/environment"

PROXY="http://192.168.5.58:1080"
export HTTP_PROXY=${PROXY}
export HTTPS_PROXY=${PROXY}
export http_proxy=${PROXY}
export https_proxy=${PROXY}
export no_proxy="localhost,127.0.0.1,::1,mirrors.tuna.tsinghua.edu.cn"
export NO_PROXY="localhost,127.0.0.1,::1,mirrors.tuna.tsinghua.edu.cn"

export LIBTORCH_BASE_DIR=/opt/torch
export LIBTORCH=${LIBTORCH_BASE_DIR}/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib

# Rust Related
CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
# Set up the cargo mirror
RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup


# Start SSHD
echo "Starting SSHD..."
exec /usr/sbin/sshd -D

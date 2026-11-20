#!/bin/bash

echo "Setting up DNN-safe GPU environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="/eos/user/${USER::1}/${USER}/ml-gpu-env"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: Python virtual environment not found: ${VENV_DIR}"
    return 1 2>/dev/null || exit 1
fi

echo "Activating Python virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "Configuring TensorFlow GPU runtime paths..."
CUDA_LIB="/cvmfs/sft.cern.ch/lcg/releases/cuda/12.3-555de/x86_64-el9-gcc11-opt/targets/x86_64-linux/lib"
CUDNN_LIB="/cvmfs/sft.cern.ch/lcg/releases/cudnn/8.9.7.29-8bd79/x86_64-el9-gcc13-opt/lib"

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${CUDA_LIB}:${CUDNN_LIB}"
fi
unset LD_PRELOAD
unset TF_USE_LEGACY_KERAS

export PATH="${REPO_DIR}:${REPO_DIR}/scripts:${REPO_DIR}/hzgml:${PATH}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/scripts:${REPO_DIR}/hzgml:${VENV_DIR}/lib/python3.12/site-packages"
export THEANO_FLAGS="gcc.cxxflags='-march=core2'"

echo "DNN-safe environment setup complete."

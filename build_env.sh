#!/bin/bash

# example for CUDA 12.4
export CUDA_HOME="/usr/local/cuda-12.4"
nvcc -V

# Install PyTorch with the specified CUDA version
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install other packages
pip install vllm==0.5.3.post1
pip install nltk
pip install --upgrade transformers
pip install tiktoken
pip install sentencepiece
pip install protobuf
pip install ninja einops triton packaging

# install Mamba
pip install -e .

# Clone and install causal-conv1d with specified CUDA version
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" python setup.py install
cd ..

echo "Installation completed."

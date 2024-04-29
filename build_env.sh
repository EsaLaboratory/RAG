#!/bin/bash

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=anaconda/envs/rag_michelon
conda create --prefix $CONPREFIX

# Activate your environment
source activate $CONPREFIX

# Install packages...
conda install -y numpy
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y conda-forge::keras
conda install -y conda-forge::tensorflow
conda install -y conda-forge::beautifulsoup4
conda install -y conda-forge::pandas
conda install -y conda-forge::matplotlib
conda install -y huggingface::transformers
conda install -y conda-forge::langchain-community
conda install -y conda-forge::langchain

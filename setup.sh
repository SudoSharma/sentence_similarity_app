#!/bin/bash
echo "Setting up CPU environment..."

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export CONDA="miniconda3"
export PATH="$HOME/$CONDA/bin:$PATH"
echo "Setting up 'sentence_similarity_app' conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n sentence_similarity_app python=3.6
source activate sentence_similarity_app 

# Install environment requirements
echo "Installing environment requirements..."
pip install --upgrade pip
conda install pytorch-cpu -c pytorch -q
pip install dill -q
pip install torchtext -q
pip install six -q
pip install flask -q
pip install flask_restful -q

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate sentence_similarity_app'."

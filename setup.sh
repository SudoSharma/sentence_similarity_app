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
conda install cython -q
pip install Flask -q
pip install flask_restful -q
conda install plac -q
conda install pytorch-cpu -c pytorch -q
conda install dill -q
pip install torchtext
conda install spacy -q
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz"
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate sentence_similarity_app'."

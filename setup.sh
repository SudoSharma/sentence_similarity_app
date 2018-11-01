#!/bin/bash
echo "Setting up CPU environment..."

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export PATH="$HOME/miniconda3/bin:$PATH"
echo "Setting up 'bimpm_app' conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n bimpm_app python=3.6
source activate bimpm_app 

# Install environment requirements
echo "Installing environment requirements..."
pip install --upgrade pip
conda install cython -q
conda install plac -q
conda install pytorch-cpu -c pytorch -q
pip install tensorboardX
conda install dill -q
pip install torchtext
conda install spacy -q
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz"
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate bimpm_app'."

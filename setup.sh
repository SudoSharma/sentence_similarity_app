#!/bin/bash
echo "Setting up CPU environment..."

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export PATH="$HOME/miniconda3/bin:$PATH"
echo "Setting up 'bimpm_app' conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n bimpm_app python=3.6
source activate bimpm_app 

# Remove en_core_sm_md from requirement.
sed -i "/en-core/ d" requirements.txt  # Remove model download

# Install environment requirements
echo "Installing environment requirements..."
pip install -q -r requirements.txt

# link spacy
conda install spacy
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz"
conda install pytorch-cpu -c pytorch --yes -q
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate bimpm_app'."

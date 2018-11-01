#!/bin/bash
echo "Setting up CPU environment..."

# Be sure to change the 'miniconda3' to your flavor of Anaconda distribution
export PATH="$HOME/miniconda3/bin:$PATH"
echo "Setting up 'bimpm_app' conda environment..."
conda config --set always_yes yes
conda update -q conda
conda create -q -n bimpm_app python=3.6
source activate bimpm_app 

# Handle spacy installation. Commenting these lines out because 'requirements.txt' has already been processed.
# sed -i "/en-core/ d" requirements.txt  # Remove model download
# echo "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz" >> requirements.txt

# Install environment requirements
echo "Installing environment requirements..."
pip install -q -r requirements.txt

# link spacy
python -m spacy link en_core_web_sm en --force

echo "Successfully installed environment!"
echo "Activate your environment with 'source activate bimpm_app'."

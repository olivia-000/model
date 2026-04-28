# conda create -n firetruck-detection python=3.8
# conda activate firetruck-detection
conda create -n tmp python=3.8
conda activate tmp
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # cu118, cu124
pip install -r requirements.txt
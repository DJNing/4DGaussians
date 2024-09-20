conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install -U 'sapien>=3.0.0b1'
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
export CUDA_HOME=$CONDA_PREFIX
# pip install -r requirements.txt
# pip install -e submodules/depth-diff-gaussian-rasterization
# pip install -e submodules/simple-knn
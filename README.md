# pytorch-path-tracing
PyTorch-based Monte Carlo Path Tracing Algorithms and More


## Installation

```bash
sudo apt install git
git clone https://github.com/Mephisto405/pytorch-path-tracing.git

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

conda create --name pypt -y python=3.12
conda activate pypt
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
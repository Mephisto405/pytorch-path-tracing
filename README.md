# PyTorch Path Tracing

Pure PyTorch-based & Vectorized Monte Carlo Path Tracing Algorithms and More for Fun

## Features
- **Path Tracing & Neural Radiosity [Hadadan et al. 2021]**: A 350-line runnable Python notebook is available [here](notebooks/neural_radiosity.ipynb).
    - **Modification**: Implemented stop-gradient on the right-hand side of the NR for better performance.

## Installation

### Step 1: Clone the Repository
First, ensure you have `git` installed. If not, you can install it with the following command:
```bash
sudo apt install git
```

Then, clone this repository:
```bash
git clone https://github.com/Mephisto405/pytorch-path-tracing.git
```

### Step 2: Set Up Conda Environment
If you don't already have Conda installed, you can use Miniforge as a lightweight alternative to Anaconda. Follow these steps to install Miniforge:

1. Download the latest Miniforge installer:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

2. Run the installer:
```bash
bash Miniforge3-Linux-x86_64.sh
```

### Step 3: Create and Activate Conda Environment
Navigate to the cloned repository directory:
```bash
cd pytorch-path-tracing
```

Create a Conda environment using the provided `environment.yaml` file:
```bash
conda create --name pypt --file environment.yaml
```

Activate the newly created environment:
```bash
conda activate pypt
```

### Step 4: Verify Installation
Ensure all dependencies are installed correctly by checking the Conda environment:
```bash
conda list
```

## Usage
Instructions on how to use the various features of the repository will be added here.

## Contributing
Feel free to contribute to this project by creating pull requests, submitting issues, or forking the repository.

## License
This project is licensed under the GPL-3.0 License, a free but copyleft license. See the LICENSE file for details.
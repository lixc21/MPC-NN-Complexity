# MPC-NN-Complexity

### Prerequisites

Ensure you have the following installed:

- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Then, create a new conda environment by `environment.yml`:

    conda env create -f environment.yml

### Run the Project

    conda activate nn-mpc
    python NNMPC.py
    python AutoTrainNNMPC.py

The result will be saved in the `data` directory.

## Set up the environment

1. Create the environment:

```bash
conda create -n "3d_face_feats" python=3.8
conda activate "3d_face_feats"
```

2. Intall PyTorch
 
First determine which version of CUDA you have installed (you can do this by running the command nvidia-smi in your terminal). Next, visit the PyTorch official website and follow the <a href="https://pytorch.org/get-started/locally/">installation guide</a> to install PyTorch with CUDA support. For example, if you have CUDA 11.7:

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install packages and download assets:

```bash
pip install -r install/requirements.txt
pip install -e src/pytorch3d/
sh install/download.sh
```

> **Note:** Since installing pytorch3d with pip can be pretty tricky, we decided to build it from source.
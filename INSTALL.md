## Set up the environment

1. Create the environment:

```bash
conda create -n "3d_face_feats" python=3.8
conda activate "3d_face_feats"
```

2. Intall PyTorch
 
First determine which version of CUDA you have installed (you can do this by running the command ```nvcc --version``` in your terminal). Next, visit the PyTorch official website and follow the <a href="https://pytorch.org/get-started/locally/">installation guide</a> to install PyTorch with CUDA support. For example, if you have CUDA 11.7:

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install PyTorch3D

Unfortunately, installing PyTorch3D can be pretty tricky. If the following commands does not work for you, please follow the instructions on their <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"> official website </a> for more details.

```bash
sh install/pytorch3d.sh
```

4. Install other packages and download assets:

```bash
pip install -r install/requirements.txt
sh install/download.sh
```
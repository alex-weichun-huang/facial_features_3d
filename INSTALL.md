## Set up the environment

1. Create the environment:

```bash
conda create -n "face_feats_3d" python=3.8
conda activate "face_feats_3d"
```

2.  Install PyTorch3D

Installing PyTorch3D can be pretty tricky. Please follow the instructions on their <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"> official website </a> for more details.


3. Install other packages and download assets:

```bash
pip install -r install/requirements.txt
sh install/download.sh
```
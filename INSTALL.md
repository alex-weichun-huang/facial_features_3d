## Set up the environment

1. Make sure mamba is installed (assuming conda is already installed): 

```bash
conda install mamba -n base -c conda-forge
```

<br>

2. Create the environment:

```bash
mamba create -n "3d_face_feats" python=3.8
mamba env update -n "3d_face_feats" --file install/conda_env.yaml
```

<br>

3. Install packages and download assets:

```bash
conda activate "3d_face_feats"
sh install/script.sh
```
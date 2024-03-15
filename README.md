# 3D_Face_Feats

Extract "3D Facial Expression Features" from video datasets using SOTA 3D face reconstruction model - EMOCA. 
The final feature will be in this format for a video:

```
feat_file = {
    "video_path": video_path,
    "label": label,
    "traj": [
        # trajectory 1
        [   
            # frame 1
            {   
                'image_path': image_path,
                'face_exp_feats': face_exp_feats,
                'frame_ind': frame_ind,
            },
            ...
        ],
        # trajectory 2
        [  
            ...
        ]
    ],
}
```

## Installation

* Follow [INSTALL.md](INSTALL.md) for installing necessary dependencies and compiling the code.

## Instructions

1. Generate a CSV file containing (video_path, video_label) on each row. Please refer to the examples under this [folder](examples/) for more information.
    
    > **Note:** We recommend mapping labels from individual dataset to the common labels defined [here](examples/common.py) so that it will be easier to merge different datasets.

        registered_emotions = {
            "Angry": 0,
            "Disgust": 1,
            "Fear": 2,
            "Happy": 3,
            "Neutral": 4,
            "Sad": 5,
            "Surprise": 6,
            "Ambiguous": 7
        }


2. Extract EMOCA features from the videos:

```sh
python run_extraction_video.py --feature_type "vals" --clip_info "./examples/dfew.csv" --output_folder "dfew_feats" --model_name "EMOCA_v2_lr_mse_20    
```

* (clip_info): The path to the CSV file you generated in Step 1.

* (dataset_type): Set it to "video" if your dataset contains mp4 files. Set it to "images" if your video dataset contains folders of images.

* (feature_type): Set it to "vis" only if your goal is to get the 3D reconstruction face clips for sanity check/ visualization purpose.

    > **Note:** The "vis" feature is way larger than the "vals" feature. We DO NOT recommend extract "vis" features for the entire dataset.

* (detect): turn on this flag if the faces in your data is not cropped out and will need to run the facial detection model.

    > **Note:** If this flag is not turned on, we assume that the faces are cropped in a way that EMOCA can use. If you are unsure about it, we recommend extracting "vis" feature from some videos and visualize for sanity check before running on the entire dataset. 
    
## Notes on EMOCA features

During inference time, EMOCA returns 2 different dictionaries:
    
1. vis:  ['inputs', 'landmarks_predicted', 'output_images_coarse', 'geometry_coarse', 'geometry_detail','mask', 'albedo', 'output_images_detail', 'uv_detail_normals', 'uv_texture_gt']

    * Values we are using includes 'expcode', 'detailcode', and 'posecode[3:6]'.

        > **Note:** The 6 dimension of posecode corresponds to [head_pitch, head_yaw, head_row, jaw_pitch, jaw_yaw, jaw_roll]. Hence, for facial expression recogntion, we are only including the jaw codes.

2. vals: ['shapecode', 'texcode', 'expcode', 'posecode', 'cam', 'lightcode', 'detailcode', 'detailemocode', 'images', 'original_code', 'predicted_images', 'predicted_detailed_image', 'predicted_translated_image', 'ops', 'normals', 'mask_face_eye', 'verts', 'albedo', 'landmarks2d', 'landmarks3d', 'predicted_landmarks', 'predicted_landmarks_mediapipe', 'trans_verts', 'masks', 'predicted_detailed_translated_image', 'translated_uv_texture', 'uv_texture_gt', 'uv_texture', 'uv_detail_normals', 'uv_shading', 'uv_vis_mask', 'uv_mask', 'uv_z', 'displacement_map']

    * Values we are using includes 'geometry_coarse', 'geometry_detail', 'output_images_coarse', and     output_images_detail'. 

## References

This directory is built on and heavily inspired by the <a href="https://github.com/radekd91/emoca">EMOCA </a> directory.

```
@inproceedings{EMOCA:CVPR:2021,
  title = {{EMOCA}: {E}motion Driven Monocular Face Capture and Animation},
  author = {Danecek, Radek and Black, Michael J. and Bolkart, Timo},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {20311--20322},
  year = {2022}
}

@article{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics (ToG), Proc. SIGGRAPH},
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}

@article{filntisis2022visual,
  title = {Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos},
  author = {Filntisis, Panagiotis P. and Retsinas, George and Paraperas-Papantoniou, Foivos and Katsamanis, Athanasios and Roussos, Anastasios and Maragos, Petros},
  journal = {arXiv preprint arXiv:2207.11094},
  publisher = {arXiv},
  year = {2022},
}
```
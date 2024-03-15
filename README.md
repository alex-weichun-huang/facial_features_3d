# 3D_Face_Feats

Extract 3D Facial Expression Features from video datasets using the state-of-the-art (SOTA) 3D face reconstruction model, EMOCA. For each video, the extracted features are organized as follows:

```
feat_file = {
    "video_path": "<path_to_video>",
    "label": "<emotion_label>",
    "traj": [
        # Trajectory 1
        [   
            # Frame 1
            {   
                'image_path': "<path_to_frame_image>",
                'face_exp_feats': "<facial_expression_features>",
                'frame_ind': "<frame_index>",
            },
            # Additional frames...
        ],
        # Additional trajectories...
    ],
}
```

## Installation

* Follow [INSTALL.md](INSTALL.md) for installing necessary dependencies and compiling the code.

## Instructions

1. Prepare a CSV file listing video paths and labels. Each row should contain a video_path and a video_label. Refer to the examples in the [datasets folder](datasets/) for guidance.

    > **Note:** We suggest mapping labels from your dataset to the common labels defined [here](examples/common.py) to simplify dataset integration.

    ```
        registered_emotions = {
            # common emotions
            "Angry": 0,
            "Disgust": 1,
            "Fear": 2,
            "Happy": 3,
            "Neutral": 4,
            "Sad": 5,
            "Surprise": 6,
            
            # dataset specific emotions
            "Ambiguous": 7,
            "Contempt": 8,
        }
    ```


2. Extract EMOCA features from your videos. Specify the path to your configuration file using the following command:

    ```sh
    python main.py --config cfg/lsvd.yaml
    ```

    > **Note:** Please check out the full configuration [here](src/config.py). We have only highlighted the more import ones here.
    
    * "clip_info": Path to the CSV file created in Step 1.
    
    * "dataset_type": Set to "video" for mp4 files or "images" for datasets with image folders.
    
    * "detect": turn on this flag if the faces in your data is not cropped out and will need to run the facial detection model.
    
      > **Note:**     If detect is not enabled, it is assumed that faces are pre-cropped for EMOCA. For best practices, consider extracting "vis" features from a subset for visual inspection before         processing the entire dataset. See [vis.ipynb](vis.ipynb) for an example. 
    
    * "feature_type": Choose "vis" as the feature type only for visual validation or inspection purposes.
    
      > **Note:**     "vis" features are significantly larger than "vals" features. Extracting "vis" features for the entire dataset is not recommended.

    
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

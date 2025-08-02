import sys
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import webdataset as wds
from src import (
    test,
    load_model,
    ImageDataset,
    VideoDataset,
    VideoBytesDataset,
    load_config,
)


def main_worker(cfg):
    

    Path(cfg['io']['output_folder']).mkdir(parents=True, exist_ok=True) 


    print(f"[Output Dir]: {cfg['io']['output_folder']}")     
    
    
    emoca, _ = load_model(cfg['emoca']['path_to_models'], cfg['emoca']['model_name'], 'detail')
    emoca.eval().to(cfg['device'])
    
    
    no_face_detected = []
    dataset = wds.WebDataset(cfg['io']['shard_path']).decode().to_tuple("mp4", "__key__")

    for video_i, (video_bytes,video_key) in enumerate(tqdm(dataset, desc="Processing shards")):
        print(f"[Processing video {video_i:06d}]: {video_key}")
        if not isinstance(video_bytes, (bytes, bytearray)) or len(video_bytes) == 0:
            warnings.warn(f"[WARNING] video_bytes is invalid or empty, got type={type(video_bytes)}, len={len(video_bytes)}. Skipping this video.")
            continue
        # load the dataset for this video
        if cfg["feature"]['dataset_type'] == 'video':
            face_dataset = VideoBytesDataset(
                video_bytes=video_bytes, 
                downsample_rate=cfg['feature']['ds_rate'],
                detect=cfg['face_detection']['detect'],
                crop_size=cfg['face_detection']['crop_size'],
                device=cfg['device'],
                iou_treshold=cfg['face_detection']['iou_threshold'],
                face_detect_thres=cfg['face_detection']['threshold']
            )
        else:
            NotImplemented



        traj = [[],]
        for img_dict in face_dataset:
            
            # handle the case when there is no face detected in this frame
            if 'image' not in img_dict:
                if cfg["feature"]['dataset_type'] == 'video':
                    traj.append([])
                    continue
                elif cfg["feature"]['dataset_type'] == 'images':
                    no_face_detected.append(img_dict['image_path'])
                    continue
            
            # handle the case when this is a new trajectory; this is only used for (videos input + detect=True)
            if "new_traj" in img_dict and img_dict["new_traj"] and len(traj[-1]) > 0:
                traj.append([])
            
            
            # get the features for this frame
            vals, vis = test(emoca, img_dict, device=cfg['device'])
            if cfg['feature']['feature_type'] == 'vis':
                frame_dict = {
                    'image_path': img_dict['image_path'],
                    'frame_ind': img_dict['frame_ind'],
                    'geometry_detail': vis['geometry_detail'].detach().cpu(),
                    # 'geometry_coarse': vis['geometry_coarse'].detach().cpu(),
                    # 'output_images_detail': vis['output_images_detail'].detach().cpu(),
                    # 'output_images_coarse': vis['output_images_coarse'].detach().cpu(),
                }
            else:
                shape_feat = vals['shapecode'][0].detach().cpu() # 50
                cam_feat = vals['cam'][0].detach().cpu() # 3
                tex_feat = vals['texcode'][0].detach().cpu() # 50
                light_feat = vals['lightcode'][0].detach().cpu().reshape(-1) # 27
                pose_feat = vals['posecode'][0].detach().cpu() # 6
                exp_feat = vals['expcode'][0].detach().cpu() # 50
                detail_feat = vals['detailcode'][0].detach().cpu() # 128
                face_exp_feats = torch.cat([
                    shape_feat, cam_feat, tex_feat, light_feat, pose_feat, exp_feat, detail_feat
                ]).numpy()
                assert face_exp_feats.shape == (364,), f'face_exp_feats.shape = {face_exp_feats.shape} instead of (364,)'

                # add resolution brightness and pose for pose filtering
                cropped_size = img_dict['cropped_size']
                n_pixel = cropped_size[0] * cropped_size[1]
                pose_score = vals["posecode"][0][1]
                brightness = get_brightness(img_dict['original_image'])

                frame_dict = {
                    'face_exp_feats': face_exp_feats,
                    'frame_ind': img_dict['frame_ind'],
                    'n_pixel': n_pixel,
                    'pose_score': pose_score.item(),
                    'brightness': brightness,
                }
            traj[-1].append(frame_dict)
        
        # save the result for this video
        feat_file = {
            'video_name': video_key,
            'traj': traj,
        }
        np.save(os.path.join(cfg['io']['output_folder'], f"{cfg['feature']['feature_type']}_{video_i:06d}.npy"), feat_file)

def get_brightness(pil_image, dim=10):
    image = np.array(pil_image.convert("RGB"))  # Ensure it's RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    image = cv2.resize(image, (dim, dim))
    L, _, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    score = np.mean(L)
    return score

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to the config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    return cfg
    
    
if __name__ == '__main__':
    cfg = parse_args()
    main_worker(cfg)
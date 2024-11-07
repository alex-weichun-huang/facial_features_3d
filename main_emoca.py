import sys
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src import (
    test,
    load_model,
    ImageDataset,
    VideoDataset,
    load_config,
)


def main_worker(cfg):
    

    Path(cfg['io']['output_folder']).mkdir(parents=True, exist_ok=True) 
    with open(cfg['io']['clip_info'], 'r') as f:
        input_videos = f.readlines()
        clip_info = [line.strip().split(',') for line in input_videos]


    print(f"[Output Dir]: {cfg['io']['output_folder']}")     
    print(f"[TODO]: {len(clip_info)}\n")
    
    
    emoca, _ = load_model(cfg['emoca']['path_to_models'], cfg['emoca']['model_name'], 'detail')
    emoca.eval().to(cfg['device'])
    
    
    no_face_detected = []
    for video_i, (video_path, label) in enumerate(tqdm(clip_info)):
        
        # load the dataset for this video
        if cfg["feature"]['dataset_type'] == 'video':
            face_dataset = VideoDataset(
                video_path=video_path, 
                downsample_rate=cfg['feature']['ds_rate'],
                detect=cfg['face_detection']['detect'],
                crop_size=cfg['face_detection']['crop_size'],
                device=cfg['device'],
                iou_treshold=cfg['face_detection']['iou_threshold'],
                face_detect_thres=cfg['face_detection']['threshold']
            )
        elif cfg["feature"]['dataset_type'] == 'images':
            face_dataset = ImageDataset(
                video_path, 
                device=cfg['device'],
                detect=cfg['face_detection']['detect'],
                crop_size=cfg['face_detection']['crop_size'], 
                face_detect_thres=cfg['face_detection']['threshold']
            )


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
                
                frame_dict = {
                    'image_path': img_dict['image_path'],
                    'face_exp_feats': face_exp_feats,
                    'frame_ind': img_dict['frame_ind'],
                }
            traj[-1].append(frame_dict)
        
        # save the result for this video
        feat_file = {
            'video_path': video_path,
            'label': label,
            'traj': traj,
        }
        np.save(os.path.join(cfg['io']['output_folder'], f"{cfg['feature']['feature_type']}_{video_i:05d}.npy"), feat_file)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to the config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    return cfg
    
    
if __name__ == '__main__':
    cfg = parse_args()
    main_worker(cfg)
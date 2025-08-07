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
from torch.utils.data import DataLoader
from src import (
    test,
    batch_test,
    load_model,
    ImageDataset,
    VideoDataset,
    VideoBytesDataset,
    OpenFaceDataset,
    load_config,
)


def main_worker(cfg):
    

    Path(cfg['io']['output_folder']).mkdir(parents=True, exist_ok=True) 

    print(f"[Input Dir]: {cfg['io']['shard_path']}")
    print(f"[Output Dir]: {cfg['io']['output_folder']}")     
    
    
    emoca, _ = load_model(cfg['emoca']['path_to_models'], cfg['emoca']['model_name'], 'detail')
    emoca.eval().to(cfg['device'])
    
    
    no_face_detected = []
    dataset = wds.WebDataset(cfg['io']['shard_path']).decode().to_tuple("npy", "__key__")

    for video_i, (arr,video_key) in enumerate(tqdm(dataset, desc="Processing shards")):
        print(f"[Processing video {video_i:06d}]: {video_key}")
        if not isinstance(arr, np.ndarray):
            warnings.warn(f"[WARNING]  invalid npy. Skipping this video.")
            continue
        face_dataset = OpenFaceDataset(arr=arr)
        batch_size = cfg['feature']['batch_size']
        face_loader = DataLoader(
            face_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        traj = []
        for batch in face_loader:
            bs = len(batch['image'])
            
            images = batch['image'].to(cfg['device'])
            frame_inds = batch['frame_ind']
            brightnesses = batch['brightness']
            
            # get the features for this frame
            with torch.no_grad():
                vals, vis = batch_test(emoca, images, device=cfg['device'])
            for i in range(bs):
                shape_feat = vals['shapecode'][i].detach().cpu()
                cam_feat = vals['cam'][i].detach().cpu()
                tex_feat = vals['texcode'][i].detach().cpu()
                light_feat = vals['lightcode'][i].detach().cpu().reshape(-1)
                pose_feat = vals['posecode'][i].detach().cpu()
                exp_feat = vals['expcode'][i].detach().cpu()
                detail_feat = vals['detailcode'][i].detach().cpu()

                face_exp_feats = torch.cat([
                    shape_feat, cam_feat, tex_feat, light_feat, pose_feat, exp_feat, detail_feat
                ]).numpy()

                assert face_exp_feats.shape == (364,), f"Got shape {face_exp_feats.shape} != (364,)"

                pose_score = pose_feat[1].item()  # yaw

                frame_dict = {
                    'face_exp_feats': face_exp_feats,
                    'frame_ind': frame_inds[i].item(),
                    'pose_score': pose_score,
                    'brightness': brightnesses[i].item(),
                    #'geometry_detail': vis['geometry_detail'][i].detach().cpu(),
                    #'image_detail': vis['output_images_detail'][i].detach().cpu(),
                }
                traj.append(frame_dict)


        
        # save the result for this video
        np.save(os.path.join(cfg['io']['output_folder'], f"{cfg['feature']['feature_type']}_{video_key}.npy"), traj)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to the config file')
    parser.add_argument('--shard_path', type=str, default=None, help='path to the shard')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.shard_path:
        data_path = os.path.expandvars("${DATA_PATH}")
        shard_path = os.path.join(data_path, args.shard_path)
        cfg['io']['shard_path'] = shard_path 
    return cfg
    
    
if __name__ == '__main__':
    cfg = parse_args()
    main_worker(cfg)

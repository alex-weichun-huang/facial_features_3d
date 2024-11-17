import sys
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
from pathlib import Path
from src import (
    VideoPILDataset,
    load_config,
)

# facetorch
from omegaconf import OmegaConf
from facetorch import FaceAnalyzer

def main_worker(cfg):
    
    Path(cfg['io']['output_folder']).mkdir(parents=True, exist_ok=True) 
    with open(cfg['io']['clip_info'], 'r') as f:
        input_videos = f.readlines()
        clip_info = [line.strip().split(',') for line in input_videos]

    print(f"[Output Dir]: {cfg['io']['output_folder']}")     
    print(f"[TODO]: {len(clip_info)}\n")
    
    # load the model
    os.makedirs("img", exist_ok=True)
    f_cfg = OmegaConf.load("cfg/gpu.config.yml")
    analyzer = FaceAnalyzer(f_cfg.analyzer)
    for video_i, (video_path, label) in enumerate(tqdm(clip_info)):
        
        if os.path.exists(os.path.join(cfg['io']['output_folder'], f"{cfg['feature']['feature_type']}_{video_i:05d}.npy")):
            print("skipping...")
            continue
        
        # load the dataset for this video
        face_dataset = VideoPILDataset(
            video_path=video_path, 
            downsample_rate=cfg['feature']['ds_rate'],
            detect=cfg['face_detection']['detect'],
            crop_size=cfg['face_detection']['crop_size'],
            device=cfg['device'],
            iou_treshold=cfg['face_detection']['iou_threshold'],
            face_detect_thres=cfg['face_detection']['threshold'],
            scale=2
        )
        
        traj = [[],]
        for img_dict in face_dataset:
            
            # handle the case when there is no face detected in this frame
            if 'image' not in img_dict:
                if cfg["feature"]['dataset_type'] == 'video':
                    traj.append([])
                    continue
            
            # handle the case when this is a new trajectory; this is only used for (videos input + detect=True)
            if "new_traj" in img_dict and img_dict["new_traj"] and len(traj[-1]) > 0:
                traj.append([])

            in_image_path = f"img/in_{cfg['run_name']}.jpg"
            out_image_path = f"img/out_{cfg['run_name']}.jpg"
            img_dict['image'].save(in_image_path)
            response = analyzer.run(
                tensor = in_image_path , 
                return_img_data=True,
                include_tensors=True
            )
            if len(response.faces) != 1:
                continue
            
            pil_image = torchvision.transforms.functional.to_pil_image(response.img)
            pil_image.save(out_image_path)
           
            
            outputs = response.faces[0]
            face_feats = {
                'au' :  outputs.preds["au"].logits.cpu().numpy(),
                'valence': outputs.preds["va"].other["valence"],
                'arousal': outputs.preds["va"].other['arousal']
            }
            frame_dict = {
                'image_path': img_dict['image_path'],
                'face_exp_feats': face_feats,
                'frame_ind': img_dict['frame_ind'],
            }
            # print(outputs.preds["au"].other["multi"])
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
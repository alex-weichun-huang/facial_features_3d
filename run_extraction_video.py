import warnings
warnings.filterwarnings("ignore")

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
)


def main_worker(args):
    
    # check the input arguments
    assert args.output_folder != None and args.clip_info != None, "Output folder and clip info must be specified."
    Path(args.output_folder).mkdir(parents=True, exist_ok=True) 
    with open(args.clip_info, "r") as f:
        input_videos = f.readlines()
        clip_info = [line.strip().split(",") for line in input_videos]

    # start the extraction
    print(f"[Output Dir]: {args.output_folder}")     
    print(f"[TODO]: {len(clip_info)}\n")
    emoca, _ = load_model(args.path_to_models, args.model_name, "detail")
    emoca.eval().to(args.device)
    
    # TODO: add support for the video dataset
    if args.dataset_type == "video":
        print("WARNING: VideoDataset is not fully tested yet.")
    
    # iterate over the videos and extract the features
    no_face_detected = []
    for video_i, (video_path, label) in enumerate(tqdm(clip_info)):
        
        # load the dataset for this video
        print(f"[Processing]: {video_path}")
        if args.dataset_type == "video":
            face_dataset = VideoDataset(video_path=video_path, face_detector_threshold=args.face_detector_threshold, downsample_rate=args.downsample_rate)
        elif args.dataset_type == "images":
            face_dataset = ImageDataset(video_path, detect=args.detect, crop_size=args.crop_size, face_detect_thres=args.face_detect_thres)

        
        traj = [[],]
        for img_dict in face_dataset:
            
            # handle the case when there is no face detected in this frame
            if "image" not in img_dict:
                if args.dataset_type == "video":
                    traj.append([])
                    continue
                elif args.dataset_type == "images":
                    no_face_detected.append(img_dict['image_path'])
                    continue
                
            vals, vis = test(emoca, img_dict, device=args.device)
            if args.feature_type == "vis":
                vis_dict = {
                    'image_path': img_dict['image_path'],
                    'output_images_detail': vis['output_images_detail'].detach().cpu(),
                    'output_images_coarse': vis['output_images_coarse'].detach().cpu(),
                    'geometry_detail': vis['geometry_detail'].detach().cpu(),
                    'geometry_coarse': vis['geometry_coarse'].detach().cpu(),
                    'frame_ind': img_dict['frame_ind'],
                }
                traj[-1].append(vis_dict)
            else:
                exp_feat = vals['expcode'][0].detach().cpu()
                pose_feat = vals['posecode'][0, 3:6].detach().cpu() 
                detail_feat = vals['detailcode'][0].detach().cpu() 
                face_exp_feats = torch.cat([pose_feat, exp_feat, detail_feat]).numpy() 
                assert face_exp_feats.shape == (181,), f"face_exp_feats.shape = {face_exp_feats.shape} instead of (181,)"
                vals_dict = {
                    'image_path': img_dict['image_path'],
                    'face_exp_feats': face_exp_feats,
                    'frame_ind': img_dict['frame_ind'],
                }
                traj[-1].append(vals_dict)
        
        # save the results
        feat_file = {
            "video_path": video_path,
            "label": label,
            "traj": traj,
        }
        np.save(os.path.join(args.output_folder, f"{args.feature_type}_{video_i:05d}.npy"), feat_file)

    
    # save the paths to the images where no face was detected
    if len(no_face_detected) > 0:
        print(f"Warning: No face detected in {len(no_face_detected)} images. Saving the paths to {args.output_folder}/no_face_detected.txt")
        with open(os.path.join(args.output_folder, "no_face_detected.txt"), "w") as f:
            f.write(f"Input folder: {args.input_folder}\n Output folder: {args.output_folder}\n")
            f.write("No face detected in the following images:\n")
            for img_path in no_face_detected:
                f.write(img_path + "\n")

    
def parse_args():
    parser = argparse.ArgumentParser()
    
    # common arguments
    parser.add_argument('--feature_type', type=str, default="vals", choices=["vals", "vis"], help="Type of features to extract. Either regression parameter (vals) or visualization results (vis).")
    parser.add_argument("--dataset_type", type=str, default="images", choices=["video", "images"], help="Type of input to use. Either video or folder of images.")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument('--downsample_rate', type=int, default=1, help="Downsampling rate for the video.")
    
    # I/O arguments
    parser.add_argument('--clip_info', type=str, default=None, help="Path to the clip info file.")
    parser.add_argument('--output_folder', type=str, default=None, help="Output folder to save the results to.")
    
    # model arguments
    parser.add_argument('--path_to_models', type=str, default= os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/EMOCA/models"), help="Path to the folder with the model.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    
    # face detection arguments
    parser.add_argument('--detect', action='store_true', help="Whether to run the face detector.")
    parser.add_argument('--face_detect_thres', type=float, default=0.5, help="Threshold for the face detector.")
    parser.add_argument('--crop_size', type=int, default=224, help="Size of the crop for the face detector.")
    parser.add_argument('--iou_treshold', type=float, default=0.5, help="Threshold for the IOU that is used in tracking.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main_worker(args)
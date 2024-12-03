import warnings
warnings.filterwarnings('ignore')

# python imports
import os
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
import torch

# facetorch
from omegaconf import OmegaConf
from facetorch import FaceAnalyzer

# our code
from src import (
    load_model,
    decode,
    VideoDataset,
    load_config,
    save_video,
    create_horizontal_clip,
    tensor_to_pil_image,
)


def overlay_image_within_bbox(base_image, overlay_image, left_top=None, cropped_size=None):
    """
    Overlays `overlay_image` on `base_image` centered at the given `center` point.

    Parameters:
    - base_image (PIL.Image): The background image on which overlay will be placed.
    - overlay_image (PIL.Image): The image to overlay on top of the base image.
    - center (tuple): A tuple (x, y) representing the center point where overlay_image should be placed.
    - scale (float): Scaling factor for the overlay image size. Default is 1.0 (no scaling).

    Returns:
    - PIL.Image: The resulting image with the overlay.
    """
    # Ensure base image is in RGBA mode
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")

    # Ensure overlay image is in RGBA mode
    if overlay_image.mode != "RGBA":
        overlay_image = overlay_image.convert("RGBA")

    # Calculate scaled dimensions of the overlay image
    overlay_width, overlay_height = cropped_size

    # Resize the overlay image to the scaled dimensions
    resized_overlay = overlay_image.resize((overlay_width, overlay_height), Image.LANCZOS)

    # Create a copy of the base image to place the overlay on
    combined_image = base_image.copy()

    # Paste the resized overlay image onto the base image, with transparency handling
    combined_image.paste(resized_overlay, (left_top[0], left_top[1]), resized_overlay)

    return combined_image


def moving_average(vals, window_size):
    smoothed_vals = []
    for i in range(len(vals)):
        if i < window_size or i >= len(vals) - window_size:
            smoothed_vals.append(vals[i])
        else:
            avg_shapecode = sum(vals[j]['shapecode'] for j in range(i - window_size, i + window_size + 1)) / (2 * window_size + 1)
            avg_posecode = sum(vals[j]['posecode'] for j in range(i - window_size, i + window_size + 1)) / (2 * window_size + 1)
            smoothed_val = vals[i].copy()
            smoothed_val['shapecode'] = avg_shapecode
            smoothed_val['posecode'] = avg_posecode
            smoothed_vals.append(smoothed_val)
    return smoothed_vals

def main_worker(cfg):
    
    # load config
    Path(cfg['io']['output_folder']).mkdir(parents=True, exist_ok=True) 
    with open(cfg['io']['clip_info'], 'r') as f:
        input_videos = f.readlines()
        clip_info = [line.strip().split(',') for line in input_videos]

    # set params
    print(f"[Output Dir]: {cfg['io']['output_folder']}")     
    print(f"[TODO]: {len(clip_info)}\n")
    in_image_path = f"img/in_{cfg['run_name']}.jpg"
    out_image_path = f"img/out_{cfg['run_name']}.jpg"
    device = cfg['device']
    
    # load emoca
    emoca, _ = load_model(cfg['emoca']['path_to_models'], cfg['emoca']['model_name'], 'detail')
    emoca.eval().to(device)
    
    for video_i, (video_path, label) in enumerate(tqdm(clip_info)):
       
        # load the dataset for this video
        face_dataset = VideoDataset(
            video_path=video_path, 
            downsample_rate=cfg['feature']['ds_rate'],
            detect=cfg['face_detection']['detect'],
            crop_size=cfg['face_detection']['crop_size'],
            device=device,
            iou_treshold=cfg['face_detection']['iou_threshold'],
            face_detect_thres=cfg['face_detection']['threshold'],
            scale=2
        )
      
        traj = [[],]
        landmark_frames = []
        overlay_frames = []
        left_frames = []
        right_frames = []
        front_frames = []
        encoded_vals = []
        original_images = []
        left_tops = []
        cropped_sizes = []
        best_encoded_vals = None
        for i, img_dict in enumerate(face_dataset):
            
            # handle the case when there is no face detected in this frame
            if 'image' not in img_dict:
                if cfg["feature"]['dataset_type'] == 'video':
                    traj.append([])
                    continue
            
            # handle the case when this is a new trajectory; this is only used for (videos input + detect=True)
            if "new_traj" in img_dict and img_dict["new_traj"] and len(traj[-1]) > 0:
                traj.append([])
            
            # get emoca image
            emoca = emoca.to(device)
            img_dict["image"] = img_dict["image"].view(1,3,224,224).to(device)
            origin_vals = emoca.encode(img_dict, training=False)
            encoded_vals.append(origin_vals)
            original_images.append(img_dict["original_image"])

            left_tops.append(img_dict['left_top'])
            cropped_sizes.append(img_dict['cropped_size'])

            if i == len(face_dataset)//2:
                best_encoded_vals= origin_vals.copy()

        # decode the smoothed values and save the output as a video
        window_size = 3 
        smoothed_vals = moving_average(encoded_vals, window_size)
        for i, (vals, left_top, cropped_size, ori_img) in enumerate(zip(smoothed_vals, left_tops, cropped_sizes, original_images)):
            left_vals = vals.copy()
            right_vals = vals.copy()
            front_vals = vals.copy()
            _, visdict = decode(emoca, vals, training=False)

            input_image = tensor_to_pil_image(visdict['inputs'].detach().cpu())
            input_image.save(in_image_path)
            landmark_image = tensor_to_pil_image(visdict['landmarks_predicted'].detach().cpu())

            left_vals["cam"][0][0] = torch.tensor([8]).to(device) # near far
            left_vals["cam"][0][1] = torch.tensor([-0.02]).to(device) 
            left_vals["cam"][0][2] = torch.tensor([0.0]).to(device)  
            
            left_vals["posecode"][0][0] = torch.tensor([0]).to(device) # up down
            left_vals["posecode"][0][1] = torch.tensor([0.5]).to(device) # left right
            left_vals["posecode"][0][2] = torch.tensor([0]).to(device) # clockwise and counter clockwise

            # fix the shapecode to the best encoded vals
            left_vals['shapecode'] = best_encoded_vals['shapecode']
            _, left_visdict = decode(emoca, left_vals, training=False)

            # right
            right_vals["cam"][0][0] = torch.tensor([8]).to(device) # near far
            right_vals["cam"][0][1] = torch.tensor([0.02]).to(device)
            right_vals["cam"][0][2] = torch.tensor([0.0]).to(device)


            right_vals["posecode"][0][0] = torch.tensor([0]).to(device) # up down
            right_vals["posecode"][0][1] = torch.tensor([-0.5]).to(device) # left right
            right_vals["posecode"][0][2] = torch.tensor([0]).to(device) # clockwise and counter clockwise

            # fix the shapecode to the best encoded vals
            right_vals['shapecode'] = best_encoded_vals['shapecode']
            _, right_visdict = decode(emoca, right_vals, training=False)

            # front
            front_vals["cam"][0][0] = torch.tensor([8]).to(device) # near far
            front_vals["cam"][0][1] = torch.tensor([0.0]).to(device)
            front_vals["cam"][0][2] = torch.tensor([0.0]).to(device)


            front_vals["posecode"][0][0] = torch.tensor([0]).to(device) # up down
            front_vals["posecode"][0][1] = torch.tensor([0.0]).to(device) # left right
            front_vals["posecode"][0][2] = torch.tensor([0]).to(device) # clockwise and counter clockwise

            # fix the shapecode to the best encoded vals
            front_vals['shapecode'] = best_encoded_vals['shapecode']
            _, front_visdict = decode(emoca, front_vals, training=False)

            landmark_frames.append(overlay_image_within_bbox(ori_img, \
                                                             landmark_image, \
                                                             left_top=left_top, \
                                                             cropped_size= cropped_size).convert("RGB"))
            overlay_frames.append(overlay_image_within_bbox(ori_img, \
                                                            tensor_to_pil_image(visdict['output_images_detail'].detach().cpu()), \
                                                             left_top=left_top, \
                                                             cropped_size= cropped_size).convert("RGB"))
            left_frames.append(tensor_to_pil_image(left_visdict['geometry_detail'].detach().cpu()))
            right_frames.append(tensor_to_pil_image(right_visdict['geometry_detail'].detach().cpu()))
            front_frames.append(tensor_to_pil_image(front_visdict['geometry_detail'].detach().cpu()))


        video_name = os.path.basename(video_path)
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        create_horizontal_clip([original_images, landmark_frames, overlay_frames], f"lmk_{video_name}", fps)
        create_horizontal_clip([front_frames, left_frames, right_frames], f"mesh_{video_name}", fps)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfg/demo.yaml', help='path to the config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    return cfg
    
    
if __name__ == '__main__':
    cfg = parse_args()
    main_worker(cfg)
import warnings
warnings.filterwarnings('ignore')

# python imports
import os
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from pathlib import Path

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
    tensor_to_pil_image,
)

def overlay_image_within_bbox(base_image, overlay_image, center, scale=1.0):
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
    overlay_width = int(overlay_image.width * scale)
    overlay_height = int(overlay_image.height * scale)

    # Resize the overlay image to the scaled dimensions
    resized_overlay = overlay_image.resize((overlay_width, overlay_height), Image.LANCZOS)

    # Calculate the top-left coordinates to center the overlay on the given center point
    left = int(center[0] - overlay_width / 2)
    top = int(center[1] - overlay_height / 2)

    # Create a copy of the base image to place the overlay on
    combined_image = base_image.copy()

    # Paste the resized overlay image onto the base image, with transparency handling
    combined_image.paste(resized_overlay, (left, top), resized_overlay)

    return combined_image

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
    
    # load facetorch
    os.makedirs("img", exist_ok=True)
    f_cfg = OmegaConf.load("cfg/gpu.config.yml")
    analyzer = FaceAnalyzer(f_cfg.analyzer)
    
    for video_i, (video_path, label) in enumerate(tqdm(clip_info)):
       
        # load the dataset for this video
        face_dataset = VideoDataset(
            video_path=video_path, 
            downsample_rate=cfg['feature']['ds_rate'],
            detect=cfg['face_detection']['detect'],
            crop_size=cfg['face_detection']['crop_size'],
            device=device,
            iou_treshold=cfg['face_detection']['iou_threshold'],
            face_detect_thres=cfg['face_detection']['threshold']
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
            
            
            # get the features for this frame
            emoca = emoca.to(device)
            img_dict["image"] = img_dict["image"].view(1,3,224,224).to(device)
            vals = emoca.encode(img_dict, training=False)
            vals, visdict = decode(emoca, vals, training=False)
            
            frame_dict = {
                'bbox': img_dict["bbox"], # (center, size)
                'original_image': img_dict["original_image"],
                'overlay_image': visdict['output_images_detail'].detach().cpu(),
            }
        
            traj[-1].append(frame_dict)
        
        # Save the output as a mp4 file
        input_frames = []
        overlay_frames = []
        landmark_frames = []
        for _traj in traj:
            for i, frame in enumerate(_traj):
               
                # get the input image
                pil_image = frame['original_image']
                pil_image.save(in_image_path)
                input_frames.append(pil_image)
                
                # get the overlay image
                center, size = frame['bbox']
                pil_image = overlay_image_within_bbox(frame['original_image'], tensor_to_pil_image(frame['overlay_image']), center = center, scale=size/224)
                overlay_frames.append(pil_image.convert("RGB"))
                
                # get the landmark
                response = analyzer.run(
                    tensor = in_image_path , 
                    return_img_data=True,
                    include_tensors=True
                )
                if len(response.faces) != 1:
                    print("Warning! More than one face detected!")
                    continue
                pil_image = torchvision.transforms.functional.to_pil_image(response.img)
                pil_image.save(out_image_path)
                landmark_frames.append(pil_image)
        
        video_name = os.path.basename(video_path)
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        save_video(input_frames, cfg['io']['output_folder'], f"input_{video_name}", fps=fps)
        save_video(overlay_frames, cfg['io']['output_folder'], f"overlay_{video_name}", fps=fps)
        save_video(landmark_frames, cfg['io']['output_folder'], f"landmark_{video_name}", fps=fps)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to the config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    return cfg
    
    
if __name__ == '__main__':
    cfg = parse_args()
    main_worker(cfg)
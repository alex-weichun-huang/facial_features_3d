import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
from glob import glob
from skimage.transform import estimate_transform, warp
from .FAN import FAN



class ImageDataset(torch.utils.data.Dataset):
    """
    Handle Face Detection and Cropping for images
    """
    
    def __init__(self, image_dir, device="cuda", detect=True, crop_size=224, face_detect_thres=0.5):
        '''
            video_path: folder, image_list, image path, video path
        '''
        self.device = device
        self.image_paths = sorted(glob(f"{image_dir}/*.jpg") + glob(f"{image_dir}/*.png"))
         
        # face detection
        self.detect = detect
        self.face_detector = FAN(self.device, threshold=face_detect_thres)
        self.crop_size = crop_size
        self.scale = 1.25 # EMOCAP uses 1.25
       
        print(f'[Path]: {image_dir}, [Images] found {len(self.image_paths)} images')
       
    def __len__(self):
        return len(self.image_paths)
    
    def bbox2point(self, left, right, top, bottom, type='bbox', scale=1.25):
        ''' 
        bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return int(old_size*scale), center
 
    def __getitem__(self, idx):
        
        # Load image 
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path).convert('RGB')
        
        if self.detect:
            # Detect face
            img_tensor = torch.from_numpy(np.array(pil_image)) 
            bbox, bbox_type = self.face_detector.run(img_tensor)
            if len(bbox) == 0:
                return {
                    'image_path': image_path,
                }
            
            # Get the size and center of the face after cropping
            left, top, right, bottom = bbox[0]
            size, center = self.bbox2point(left, right, top, bottom, type=bbox_type, scale=self.scale)
            
            # Crop face
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            dst_pts = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, dst_pts)
            face_image = warp(img_tensor.numpy(), tform.inverse, output_shape=(self.crop_size, self.crop_size))
            face_image = (face_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(face_image)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(pil_image)) 
        img_tensor = img_tensor.permute(2, 0, 1) 
        img_tensor = img_tensor.float() / 255
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=224, mode='bilinear', align_corners=True).squeeze(0)
        
        return {
            'image': img_tensor,
            'frame_ind': idx,
            'image_path': image_path,
        }


class VideoDataset(torch.utils.data.IterableDataset):
    """
    Handle Face Detection and Cropping for videos
    """
    
    def __init__(self, video_path, downsample_rate=1, detect=True, crop_size=224, device="cuda", iou_treshold=0.5, face_detect_thres=0.5):
        '''
            video_path: folder, image_list, image path, video path
        '''
        assert os.path.isfile(video_path) and video_path[-3:] == 'mp4', f'Invalid video path: {video_path}'
          
        self.device = device
        self.downsample_rate = downsample_rate
        
        self.detect = detect
        self.crop_size = crop_size
        self.scale = 1.25 # EMOCA uses 1.25
        self.iou_threshold = iou_treshold
        self.face_detect_thres = face_detect_thres
        if self.detect:
            self.face_detector = FAN(
                self.device, 
                threshold=face_detect_thres
            )
        
        self.prev_box = None
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.ds_frame_count = self.get_downsampled_frame_count(video_path)
        print(f'[Path]: {video_path}, [Frames] found {self.ds_frame_count} frames')

    def __len__(self):
        return self.ds_frame_count
    
    
    def get_downsampled_frame_count(self, video_path):
        # NOTE: This version is faster but doesn't work for trimmed videos
        return math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.downsample_rate)
    
    
    def bbox2point(self, left, right, top, bottom, type='bbox', scale=1.25):
        ''' 
        bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return int(old_size*scale), center

    
    def iou(self, pre_box, boxes):
        """ 
        Return the intersection over union (IOU) between each box in `boxes` and `pre_box`.
        """
        # Calculate coordinates of intersection rectangle
        x1 = np.maximum(pre_box[0], boxes[:, 0])
        y1 = np.maximum(pre_box[1], boxes[:, 1])
        x2 = np.minimum(pre_box[2], boxes[:, 2])
        y2 = np.minimum(pre_box[3], boxes[:, 3])
        
        # Calculate areas of intersection and union
        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        pre_box_area = (pre_box[2] - pre_box[0]) * (pre_box[3] - pre_box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = pre_box_area + boxes_area - intersection_area
        
        # Calculate IOU
        iou = intersection_area / union_area
        return iou
    
 
    def __iter__(self):
        
        index = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            index += 1
            if index % self.downsample_rate != 0:
                continue
                
            new_traj = False
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not self.detect:
                bbox = [0, 0, frame.shape[1], frame.shape[0]]
                new_traj = False # assuming the whole video is a single trajectory
                self.prev_box = bbox
            
            else:
                # Detect face
                img_tensor = torch.from_numpy(np.array(pil_image)) 
                bboxes, bbox_type = self.face_detector.run(img_tensor)
                
                # if no face detected
                if len(bboxes) == 0:
                    self.prev_box = None
                    yield {
                        'image_path': f"{self.video_path}_frame_{index}.jpg",
                        'frame_ind': index,
                        'new_traj': True,
                    }
                    continue
                
                # check if the detected face is in the same trajectory as the previous frame
                if self.prev_box is not None:
                    ious = self.iou(self.prev_box, np.array(bboxes))
                    best_match = np.argmax(ious)
                    if ious[best_match] >= self.iou_threshold:
                        bbox = bboxes[best_match]
                        self.prev_box = bbox
                    else:      
                        bbox = bboxes[0]
                        new_traj = True
                        self.prev_box = bbox
                else:
                    bbox = bboxes[0]
                    new_traj = True
                    self.prev_box = bbox

                # Get the size and center of the face after cropping
                left, top, right, bottom = bbox
                size, center = self.bbox2point(left, right, top, bottom, type=bbox_type, scale=self.scale)
                
                # Crop face
                src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
                dst_pts = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
                tform = estimate_transform('similarity', src_pts, dst_pts)
                face_image = warp(img_tensor.numpy(), tform.inverse, output_shape=(self.crop_size, self.crop_size))
                face_image = (face_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(face_image)

            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(pil_image)) 
            img_tensor = img_tensor.permute(2, 0, 1) 
            img_tensor = img_tensor.float() / 255
            img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=224, mode='bilinear', align_corners=True).squeeze(0)
            
            yield {
                'image': img_tensor,
                'image_path': f"{self.video_path}_frame_{index}.jpg",
                'frame_ind': index,
                'new_traj': new_traj,
            }
        self.cap.release()
 
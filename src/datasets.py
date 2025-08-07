import io
import os
import cv2
import av
import math
import torch
import numpy as np
from PIL import Image
from glob import glob
from skimage.transform import estimate_transform, warp
from .detectors import FAN
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image



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
    
    def __init__(self, video_path, downsample_rate=1, detect=True, crop_size=224, device="cuda", iou_treshold=0.5, face_detect_thres=0.5, scale=1.25):
        '''
            video_path: folder, image_list, image path, video path
        '''
        assert os.path.isfile(video_path) and video_path[-3:] == 'mp4', f'Invalid video path: {video_path}'
          
        self.device = device
        self.downsample_rate = downsample_rate
        
        self.detect = detect
        self.crop_size = crop_size
        self.scale = scale # EMOCA uses 1.25
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
            old_size = (right - left + bottom - top)/2.0
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size*scale, center

    
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
            og_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not self.detect:
                bbox = [0, 0, frame.shape[1], frame.shape[0]]
                new_traj = False # assuming the whole video is a single trajectory
                self.prev_box = bbox
            
            else:
                # Detect face
                og_img_tensor = torch.from_numpy(np.array(og_pil_image)) 
                bboxes, bbox_type = self.face_detector.run(og_img_tensor)
                
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
                # src_pts = np.array([[center[0]-size/2.0, center[1]-size/2], [center[0] - size/2.0, center[1]+size/2.0], [center[0]+size/2.0, center[1]-size/2.0]])
                # dst_pts = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
                # tform = estimate_transform('similarity', src_pts, dst_pts)
                # face_image = warp(og_img_tensor.numpy(), tform.inverse, output_shape=(self.crop_size, self.crop_size))
                # face_image = (face_image * 255).astype(np.uint8)
                # pil_image = Image.fromarray(face_image)
                center_x, center_y = center
                crop_left = center_x - size / 2.0
                crop_top = center_y - size / 2.0
                crop_right = center_x + size / 2.0
                crop_bottom = center_y + size / 2.0

                # Perform cropping with integer coordinates
                cropped_image = og_pil_image.crop((int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
                # size of the cropped region
                crop_width, crop_height = cropped_image.size

                # Resize the cropped region to the desired size
                cropped_image = cropped_image.resize((224, 224), Image.LANCZOS)

            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(cropped_image)) 
            img_tensor = img_tensor.permute(2, 0, 1) 
            img_tensor = img_tensor.float() / 255
            img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=224, mode='bilinear', align_corners=True).squeeze(0)
            
            yield {
                'bbox': [center, size],
                'original_image': og_pil_image,
                'image': img_tensor,
                'image_path': f"{self.video_path}_frame_{index}.jpg",
                'frame_ind': index,
                'new_traj': new_traj,
                'left_top': (int(crop_left), int(crop_top)),
                'cropped_size': (crop_width, crop_height),
            }
        self.cap.release()

class VideoPILDataset(torch.utils.data.IterableDataset):
    """
    Handle Face Detection and Cropping for videos
    """
    
    def __init__(self, video_path, downsample_rate=1, detect=True, crop_size=224, device="cuda", iou_treshold=0.5, face_detect_thres=0.5, scale=1.25):
        '''
            video_path: folder, image_list, image path, video path
        '''
        assert os.path.isfile(video_path) and video_path[-3:] == 'mp4', f'Invalid video path: {video_path}'
          
        self.device = device
        self.downsample_rate = downsample_rate
        
        self.detect = detect
        self.crop_size = crop_size
        self.scale = scale # EMOCA uses 1.25
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

            yield {
                'image': pil_image,
                'image_path': f"{self.video_path}_frame_{index}.jpg",
                'frame_ind': index,
                'new_traj': new_traj,
            }
        self.cap.release()
 
class VideoBytesDataset(torch.utils.data.IterableDataset):
    """
    Handle Face Detection and Cropping for videos loaded from in-memory bytes using PyAV
    """

    def __init__(self, video_bytes, downsample_rate=1, detect=True, crop_size=224, device="cuda", iou_treshold=0.5, face_detect_thres=0.5):
        self.device = device
        self.downsample_rate = downsample_rate
        self.detect = detect
        self.crop_size = crop_size
        self.scale = 1.25
        self.iou_threshold = iou_treshold
        self.face_detect_thres = face_detect_thres
        if self.detect:
            self.face_detector = FAN(self.device, threshold=face_detect_thres)

        self.container = av.open(io.BytesIO(video_bytes))
        self.stream = self.container.streams.video[0]
        self.prev_box = None

    def bbox2point(self, left, right, top, bottom, type='bbox', scale=1.25):
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        else:
            raise NotImplementedError
        return int(old_size * scale), center

    def iou(self, pre_box, boxes):
        x1 = np.maximum(pre_box[0], boxes[:, 0])
        y1 = np.maximum(pre_box[1], boxes[:, 1])
        x2 = np.minimum(pre_box[2], boxes[:, 2])
        y2 = np.minimum(pre_box[3], boxes[:, 3])
        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        pre_box_area = (pre_box[2] - pre_box[0]) * (pre_box[3] - pre_box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = pre_box_area + boxes_area - intersection_area
        return intersection_area / union_area

    def __iter__(self):
        index = 0
        for frame in self.container.decode(self.stream):
            index += 1
            if index % self.downsample_rate != 0:
                continue

            new_traj = False
            frame_img = frame.to_ndarray(format="bgr24")
            pil_image = Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

            if not self.detect:
                bbox = [0, 0, frame_img.shape[1], frame_img.shape[0]]
                self.prev_box = bbox
            else:
                img_tensor = torch.from_numpy(np.array(pil_image))
                bboxes, bbox_type = self.face_detector.run(img_tensor)

                if len(bboxes) == 0:
                    self.prev_box = None
                    yield {
                        'image_path': f"in_memory_frame_{index}.jpg",
                        'frame_ind': index,
                        'new_traj': True,
                    }
                    continue

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

                left, top, right, bottom = bbox
                size, center = self.bbox2point(left, right, top, bottom, type=bbox_type, scale=self.scale)
                src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0]-size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
                dst_pts = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
                tform = estimate_transform('similarity', src_pts, dst_pts)
                face_image = warp(img_tensor.numpy(), tform.inverse, output_shape=(self.crop_size, self.crop_size))
                face_image = (face_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(face_image)

            img_tensor = torch.from_numpy(np.array(pil_image))
            img_tensor = img_tensor.permute(2, 0, 1).float() / 255
            img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=224, mode='bilinear', align_corners=True).squeeze(0)

            yield {
                'bbox': [center, size],
                'original_image': pil_image,
                'image': img_tensor,
                'image_path': f"in_memory_frame_{index}.jpg",
                'frame_ind': index,
                'new_traj': new_traj,
                'cropped_size': img_tensor.shape[1:]
            }

class OpenFaceDataset(torch.utils.data.Dataset):
    def __init__(self, arr, resize_to=224, scale=1.25):
        self.frames = arr.astype(np.uint8)
        self.resize_to = resize_to
        self.scale = scale

        self.tensor_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),  # (HWC) -> (CHW), float [0,1]
        ])
        self.pil_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_to, resize_to)),
        ])

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        img = self.frames[idx]
        brightness = get_brightness_from_bgr_np_img(img) 

        # convert to RGB
        img = img[..., ::-1]  # OpenFace uses BGR, convert to RGB
        h, w, _ = img.shape
        size = int(max(h, w) * self.scale)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        offset_y = (size - h) // 2
        offset_x = (size - w) // 2
        padded[offset_y:offset_y + h, offset_x:offset_x + w] = img
        img = padded
        img_tensor = self.tensor_transform(img)  # (3, 224, 224)
        return {
            "image": img_tensor,
            "brightness": brightness,
            "frame_ind": idx,
        }

def get_brightness_from_bgr_np_img(img):

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_channel = lab_image[:, :, 0]
    non_zero_pixels = np.any(img != 0, axis=-1)
    non_zero_l_channel = l_channel[non_zero_pixels]
    if non_zero_l_channel.size == 0:
        return 0.0
    score = np.mean(non_zero_l_channel) / 255.0  # Normalize to
    return score


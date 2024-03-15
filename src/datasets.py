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
    
    def __init__(self, image_dir, crop_size=224, device="cuda", face_detect_thres=0.5, detect=True):
        '''
            testpath: folder, image_list, image path, video path
        '''
        self.device = device
        self.image_paths = sorted(glob(f"{image_dir}/*.jpg") + glob(f"{image_dir}/*.png"))
         
        # face detection
        self.detect = detect
        self.face_detector = FAN(self.device, threshold=face_detect_thres)
        self.crop_size = crop_size
        self.scale = 1.25 # EMOCAP uses 1.25
       
        print(f'[Dir]: {image_dir}, [Image] found {len(self.image_paths)} images')
       
    def __len__(self):
        return len(self.image_paths)
    
    def bbox2point(self, left, right, top, bottom, type='bbox', scale=1.25):
        ''' bbox from detector and landmarks are different
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
    For efficiency, we don't write the video frames into images in local folder and then read them back.
    Instead, we load the video and save their fames as a variable.
    """
    
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector="FAN", downsample_rate=1, device="cuda", face_detector_threshold=0.5, iou_treshold=0.5):
        '''
            testpath: folder, image_list, image path, video path
        '''
        assert os.path.isfile(testpath) and testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm', 'avi'], f'Wrong testpath: {testpath}'
          
        self.downsample_rate = downsample_rate
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.device = device
        self.iou_threshold = iou_treshold
        self.face_detector_threshold = face_detector_threshold
        self.prev_box = None
        self.cap = cv2.VideoCapture(testpath)
        self.ds_frame_count = self.get_downsampled_frame_count(testpath)
        if face_detector == 'FAN':
            self.face_detector = FAN(self.device, threshold=self.face_detector_threshold)
        else:
            raise NotImplementedError
        print(f'[Video]: {testpath}')
        print(f'[Frame]: {self.ds_frame_count}')
         

    def __len__(self):
        return self.ds_frame_count
    
    
    def get_downsampled_frame_count(self, video_path):
        # NOTE: This version is faster but doesn't work for trimmed videos
        return math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.downsample_rate)
    
    
    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    
    def iou(self, pre_box, boxes):
        """ Given two numpy arrays `pre_box` and `boxes`, return the intersection over union (IOU) between each box in `boxes` and `pre_box`.
        Args:
            pre_box:        (numpy array) [1,4] each row containing [x1,y1,x2,y2] coordinates
            boxes:          (numpy array) [N,4] each row containing [x1,y1,x2,y2] coordinates

        Returns:
            ious:           (numpy array) [N,1] each row containing the IOU between `pre_box` and the corresponding row in `boxes`
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
        
        index = -1 # start from -1 so that the first frame is 0
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            index += 1
            if index % self.downsample_rate == 0:
                
                ##### NOTE: Since we don't load raw images, to keep the same RGB channels as TestData, we convert the order #####
                image = np.array(frame)[...,::-1] 

                # get the shape of image
                if len(image.shape) == 2:
                    image = image[:,:,None].repeat(1,1,3) # gray image
                elif len(image.shape) == 3 and image.shape[2] >= 3:
                    image = image[:,:,:3]
                else:
                    raise AssertionError('The image shape is wrong!')
                h, w, _ = image.shape
                
                # get the bounding boxex
                result = self.face_detector.run(image) 
                if len(result) == 2:
                    bboxes, bbox_type = result
                elif len(result) == 1:
                    bboxes = result
                else:
                    raise AssertionError('The result of face detector is wrong!')
                
                
                # No box (face) detected
                if len(bboxes) == 0:
                    self.prev_box = None
                    yield {
                        'image': torch.tensor(image).float(),
                        'frame_ind': index,
                    }
                    continue
                
                # if we are extending the previous track
                new_traj = False 
                bbox = None
                if self.prev_box is not None:
                    ious = self.iou(self.prev_box, np.array(bboxes))
                    best_match = np.argmax(ious)
                    if ious[best_match] >= self.iou_threshold:
                        bbox = bboxes[best_match]
                        self.prev_box = bbox

                # if we are starting a new track, for simplicity we just take the first face in the frame
                if bbox is None:
                    if len(bboxes) > 1:
                        bbox = bboxes[1]
                    else:
                        bbox = bboxes[0]
                    new_traj = True
                    self.prev_box = bbox
                
                # return the warped image
                left, top, right, bottom = bbox
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                size = int(old_size * self.scale)
                src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
                dst_pts = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
                tform = estimate_transform('similarity', src_pts, dst_pts)
                
                image = image/255. # original image [height, width, 3]
                face_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
                face_image = face_image.transpose(2,0,1) # [224, 224, 3] --> [3, 224, 224]
                # cv2.imwrite(f'face_image_{index}.jpg', face_image.transpose(1,2,0)[...,::-1]*255.)
                # exit(0)
                
                yield {
                        'image': torch.tensor(face_image).float(),
                        'bbox': np.array(bbox).reshape(1,4), # (x1, y1, x2, y2)
                        'frame_ind': index,
                        'new_traj': new_traj,
                        }
        self.cap.release()
 
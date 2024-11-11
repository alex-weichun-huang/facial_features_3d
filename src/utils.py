import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image


#------------------------------------------------


def check_mkdir(path):
    if not os.path.exists(path):
        print('creating %s' % path)
        os.makedirs(path)
        

def locate_checkpoint(cfg, replace_root = None, relative_to = None, mode=None):
    checkpoint_dir = cfg.inout.checkpoint_dir
    if replace_root is not None and relative_to is not None:
        try:
            checkpoint_dir = str(Path(replace_root) / Path(checkpoint_dir).relative_to(relative_to))
        except ValueError as e:
            print(f"Not replacing the root of checkpoint_dir '{checkpoint_dir}' beacuse the specified root does not fit:"
                  f"'{replace_root}'")
    checkpoints = sorted(list(Path(checkpoint_dir).rglob("*.ckpt")))
    if len(checkpoints) == 0:
        print(f"Did not find checkpoints. Looking in subfolders")
        checkpoints = sorted(list(Path(checkpoint_dir).rglob("*.ckpt")))
        if len(checkpoints) == 0:
            print(f"Did not find checkpoints to resume from. Returning None")
            return None
    else:
        pass
    
    if isinstance(mode, int):
        checkpoint = str(checkpoints[mode])
    elif mode == 'latest':
        checkpoint = str(checkpoints[-1])
    elif mode == 'best':
        min_value = 999999999999999.
        min_idx = -1
        for idx, ckpt in enumerate(checkpoints):
            if ckpt.stem == "last": # disregard last
                continue
            end_idx = str(ckpt.stem).rfind('=') + 1
            loss_str = str(ckpt.stem)[end_idx:]
            try:
                loss_value = float(loss_str)
            except ValueError as e:
                print(f"Unable to convert '{loss_str}' to float. Skipping this checkpoint.")
                continue
            if loss_value <= min_value:
                min_value = loss_value
                min_idx = idx
        if min_idx == -1:
            raise FileNotFoundError("Finding the best checkpoint failed")
        checkpoint = str(checkpoints[min_idx])
    else:
        raise ValueError(f"Invalid checkpoint loading mode '{mode}'")
    # print(f"Selecting checkpoint '{checkpoint}'")
    return checkpoint


def hack_paths(cfg, replace_root_path=None, relative_to_path=None):
    if relative_to_path is not None and replace_root_path is not None:
        cfg.model.flame_model_path = str(Path(replace_root_path) / Path(cfg.model.flame_model_path).relative_to(relative_to_path))
        cfg.model.flame_lmk_embedding_path = str(Path(replace_root_path) / Path(cfg.model.flame_lmk_embedding_path).relative_to(relative_to_path))
        cfg.model.tex_path = str(Path(replace_root_path) / Path(cfg.model.tex_path).relative_to(relative_to_path))
        cfg.model.topology_path = str(Path(replace_root_path) / Path(cfg.model.topology_path).relative_to(relative_to_path))
        cfg.model.face_mask_path = str(Path(replace_root_path) / Path(cfg.model.face_mask_path).relative_to(relative_to_path))
        cfg.model.face_eye_mask_path = str(Path(replace_root_path) / Path(cfg.model.face_eye_mask_path).relative_to(relative_to_path))
        cfg.model.fixed_displacement_path = str(Path(replace_root_path) / Path(cfg.model.fixed_displacement_path).relative_to(relative_to_path))
        cfg.model.pretrained_vgg_face_path = str(Path(replace_root_path) / Path(cfg.model.pretrained_vgg_face_path).relative_to(relative_to_path))
        cfg.model.pretrained_modelpath = '/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar'
        if cfg.data.data_root is not None:
            cfg.data.data_root = str(Path(replace_root_path) / Path(cfg.data.data_root).relative_to(relative_to_path))
        try:
            cfg.inout.full_run_dir = str(Path(replace_root_path) / Path(cfg.inout.full_run_dir).relative_to(relative_to_path))
        except ValueError as e:
            print(f"Skipping hacking full_run_dir {cfg.inout.full_run_dir} because it does not start with '{relative_to_path}'")
    return cfg


def replace_asset_dirs(cfg, output_dir : Path): 
    
    asset_dir = Path(__file__).parent.parent / "assets"
    for mode in ["coarse", "detail"]:
        cfg[mode].inout.output_dir = str(output_dir.parent)
        cfg[mode].inout.full_run_dir = str(output_dir / mode)
        cfg[mode].inout.checkpoint_dir = str(output_dir / mode / "checkpoints")

        cfg[mode].model.tex_path = str(asset_dir / "FLAME/texture/FLAME_albedo_from_BFM.npz")
        cfg[mode].model.topology_path = str(asset_dir / "FLAME/geometry/head_template.obj")
        cfg[mode].model.fixed_displacement_path = str(asset_dir / 
                "FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy")
        cfg[mode].model.flame_model_path = str(asset_dir / "FLAME/geometry/generic_model.pkl")
        cfg[mode].model.flame_lmk_embedding_path = str(asset_dir / "FLAME/geometry/landmark_embedding.npy")
        cfg[mode].model.flame_mediapipe_lmk_embedding_path = str(asset_dir / "FLAME/geometry/mediapipe_landmark_embedding.npz")
        cfg[mode].model.face_mask_path = str(asset_dir / "FLAME/mask/uv_face_mask.png")
        cfg[mode].model.face_eye_mask_path  = str(asset_dir / "FLAME/mask/uv_face_eye_mask.png")
        cfg[mode].model.pretrained_modelpath = str(asset_dir / "DECA/data/deca_model.tar")
        cfg[mode].model.pretrained_vgg_face_path = str(asset_dir /  "FaceRecognition/resnet50_ft_weight.pkl") 
        cfg[mode].model.emonet_model_path = ""
    
    return cfg


#------------------------------------------------


def fix_image( image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def upsample_mesh(vertices, normals, faces, displacement_map, texture_map, dense_template):
    ''' upsampling coarse mesh (with displacment map)
        vertices: vertices of coarse mesh, [nv, 3]
        normals: vertex normals, [nv, 3]
        faces: faces of coarse mesh, [nf, 3]
        texture_map: texture map, [256, 256, 3]
        displacement_map: displacment map, [256, 256]
        dense_template:
    Returns:
        dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
        dense_colors: vertex color, [number of dense vertices, 3]
        dense_faces: [number of dense faces, 3]
    '''
    img_size = dense_template['img_size']
    dense_faces = dense_template['f']
    x_coords = dense_template['x_coords']
    y_coords = dense_template['y_coords']
    valid_pixel_ids = dense_template['valid_pixel_ids']
    valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
    valid_pixel_b_coords = dense_template['valid_pixel_b_coords']

    pixel_3d_points = vertices[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                    vertices[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                    vertices[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    vertex_normals = normals
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                    vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                    vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    dense_colors = texture_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    return dense_vertices, dense_colors, dense_faces


#------------------------------------------------
def save_video(frames, dir_name, filename, fps=10):
    video = cv2.VideoWriter(os.path.join(dir_name, filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, frames[0].size)
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
    print(f"Video saved to {os.path.join(dir_name, filename)}")

def create_horizontal_clip(frame_lists, filename, fps=10):
    # Ensure all frame lists have the same number of frames
    num_frames = min(len(lst) for lst in frame_lists)
    frame_width, frame_height = frame_lists[0][0].size  # Assume all frames have the same dimensions
    
    # Define the width and height for the combined frame (1x3 layout)
    combined_width = frame_width * 3
    combined_height = frame_height

    # Prepare the video writer (adjust fps and output path as needed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (combined_width, combined_height))

    # Process each frame and write it to the video
    for i in range(num_frames):
        # Stack frames horizontally
        frames = [frame_lists[j][i] for j in range(3)]
        combined_frame = Image.new("RGB", (combined_width, combined_height))
        
        # Paste each frame into the combined frame
        for j, frame in enumerate(frames):
            combined_frame.paste(frame, (j * frame_width, 0))
        
        # Convert to OpenCV format and write to video
        cv2_frame = cv2.cvtColor(np.array(combined_frame), cv2.COLOR_RGB2BGR)
        out.write(cv2_frame)
    
    # Release the video writer
    out.release()
    print("Video saved as output_video.mp4")

def dict_cuda_to_cpu(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_cuda_to_cpu(v)
        elif isinstance(v, torch.Tensor):
            d[k] = v.detach().cpu()
    return d


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def tensor_to_pil_image(tensor_image):
    """
    Convert a PyTorch tensor to a PIL image.
    """
    tensor_image = tensor_image.squeeze(0)  # Remove the batch dimension if it's there
    tensor_image = tensor_image.permute(1, 2, 0)  # Change the order of dimensions to (H, W, C)
    tensor_image = tensor_image.mul(255).byte()  # Scale to 0-255 and convert to bytes
    numpy_image = tensor_image.cpu().numpy()  # Convert back to a numpy array
    pil_image = Image.fromarray(numpy_image)  # Convert to a PIL image
    return pil_image


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()


def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y =  bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    elif type == "mediapipe":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        raise NotImplementedError(f" bbox2point not implemented for {type} ")
    if isinstance(center_x, np.ndarray):
        center = np.stack([center_x, center_y], axis=1)
    else: 
        center = np.array([center_x, center_y])
    return old_size, center
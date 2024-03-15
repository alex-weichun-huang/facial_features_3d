import os
import sys
import cv2
import torch
import adabound
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F_v
from pathlib import Path
from skimage.io import imread
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from .loss import IDMRFLoss
from .flame import FLAME_mediapipe
from .encoder import ResnetEncoder
from .renderer import SRenderY
from .generator import Generator
from .utils import (
    load_local_mask, 
    tensor2image, 
    upsample_mesh, 
    write_obj, 
    vertex_normals, 
    class_from_str, 
    batch_orth_proj, 
    tensor_vis_landmarks
)


class DECA(torch.nn.Module):
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, config):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()
        
        # ID-MRF perceptual loss (kept here from the original DECA implementation)
        self.perceptual_loss = None
        
        # Face Recognition loss
        self.id_loss = None

        # VGG feature loss
        self.vgg_loss = None
        
        self._reconfigure(config)
        self._reinitialize()

    def get_input_image_size(self): 
        return (self.config.image_size, self.config.image_size)

    def _reconfigure(self, config):
        self.config = config
        
        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        # identity-based detail code 
        self.n_detail = config.n_detail
        # emotion-based detail code (deprecated, not use by DECA or EMOCA)
        self.n_detail_emo = config.n_detail_emo if 'n_detail_emo' in config.keys() else 0

        # count the size of the conidition vector
        if 'detail_conditioning' in self.config.keys():
            self.n_cond = 0
            if 'globalpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'jawpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'identity' in self.config.detail_conditioning:
                self.n_cond += config.n_shape
            if 'expression' in self.config.detail_conditioning:
                self.n_cond += config.n_exp
        else:
            self.n_cond = 3 + config.n_exp

        self.mode = 1 if str(config.mode).upper() == "COARSE" else 2
        self._create_detail_generator()
        self._init_deep_losses()
        self._setup_neural_rendering()

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()
        self._init_deep_losses()
        self.face_attr_mask = load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _get_num_shape_params(self): 
        return self.config.n_shape

    def _init_deep_losses(self):
        """
        Initialize networks for deep losses
        """
        # TODO: ideally these networks should be moved out the DECA class and into DecaModule, 
        # but that would break backwards compatility with the original DECA and would not be able to load DECA's weights
        if 'mrfwr' not in self.config.keys() or self.config.mrfwr == 0:
            self.perceptual_loss = None
        else:
            if self.perceptual_loss is None:
                self.perceptual_loss = IDMRFLoss().eval()
                self.perceptual_loss.requires_grad_(False)  # TODO, move this to the constructor

        if 'idw' not in self.config.keys() or self.config.idw == 0:
            self.id_loss = None
        # alex: not use for now
        # else:
        #     if self.id_loss is None:
        #         id_metric = self.config.id_metric if 'id_metric' in self.config.keys() else None
        #         id_trainable = self.config.id_trainable if 'id_trainable' in self.config.keys() else False
        #         self.id_loss_start_step = self.config.id_loss_start_step if 'id_loss_start_step' in self.config.keys() else 0
        #         self.id_loss = lossfunc.VGGFace2Loss(self.config.pretrained_vgg_face_path, id_metric, id_trainable)
        #         self.id_loss.freeze_nontrainable_layers()


        if 'vggw' not in self.config.keys() or self.config.vggw == 0:
            self.vgg_loss = None
        # alex: not use for now
        # else:
        #     if self.vgg_loss is None:
        #         vgg_loss_batch_norm = 'vgg_loss_batch_norm' in self.config.keys() and self.config.vgg_loss_batch_norm
        #         self.vgg_loss = VGG19Loss(dict(zip(self.config.vgg_loss_layers, self.config.lambda_vgg_layers)), batch_norm=vgg_loss_batch_norm).eval()
        #         self.vgg_loss.requires_grad_(False) # TODO, move this to the constructor

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path,
                               uv_size=self.config.uv_size)  # .to(self.device)
        # face mask for rendering details
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # displacement mask is deprecated and not used by DECA or EMOCA
        if 'displacement_mask' in self.config.keys():
            displacement_mask_ = 1-np.load(self.config.displacement_mask).astype(np.float32)
            # displacement_mask_ = np.load(self.config.displacement_mask).astype(np.float32)
            displacement_mask_ = torch.from_numpy(displacement_mask_)[None, None, ...].contiguous()
            displacement_mask_ = F.interpolate(displacement_mask_, [self.config.uv_size, self.config.uv_size])
            self.register_buffer('displacement_mask', displacement_mask_)

        ## displacement correct
        if os.path.isfile(self.config.fixed_displacement_path):
            fixed_dis = np.load(self.config.fixed_displacement_path)
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
            print("Warning: fixed_displacement_path not found, using zero displacement")
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def uses_texture(self): 
        if 'use_texture' in self.config.keys():
            return self.config.use_texture
        return True # true by default

    def _disable_texture(self, remove_from_model=False): 
        self.config.use_texture = False
        if remove_from_model:
            self.flametex = None

    def _enable_texture(self):
        self.config.use_texture = True

    def _has_neural_rendering(self):
        return hasattr(self.config, "neural_renderer") and bool(self.config.neural_renderer)

    def _setup_neural_rendering(self):
        if self._has_neural_rendering():
            # alex: not use for now
            # if self.config.neural_renderer.class_ == "StarGAN":
            #     from .StarGAN import StarGANWrapper
            #     print("Creating StarGAN neural renderer")
            #     self.image_translator = StarGANWrapper(self.config.neural_renderer.cfg, self.config.neural_renderer.stargan_repo)
            # else:
            #     raise ValueError(f"Unsupported neural renderer class '{self.config.neural_renderer.class_}'")

            if self.image_translator.background_mode == "input":
                if self.config.background_from_input not in [True, "input"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "Background should be inpainted from the input")
            elif self.image_translator.background_mode == "black":
                if self.config.background_from_input not in [False, "black"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "Background should be black.")
            elif self.image_translator.background_mode == "none":
                if self.config.background_from_input not in ["none"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "The background should not be handled")
            else:
                raise NotImplementedError(f"Unsupported mode of the neural renderer backroungd: "
                                          f"'{self.image_translator.background_mode}'")

    def _create_detail_generator(self):
        #backwards compatibility hack:
        if hasattr(self, 'D_detail'):
            if (not "detail_conditioning_type" in self.config.keys() or  self.config.detail_conditioning_type == "concat") \
                and isinstance(self.D_detail, Generator):
                return
            
            # alex: not use for now
            # if self.config.detail_conditioning_type == "adain" and isinstance(self.D_detail, GeneratorAdaIn):
            #     return
            print("[WARNING]: We are reinitializing the detail generator!")
            del self.D_detail # just to make sure we free the CUDA memory, probably not necessary

        if not "detail_conditioning_type" in self.config.keys() or str(self.config.detail_conditioning_type).lower() == "concat":
            # concatenates detail latent and conditioning (this one is used by DECA/EMOCA)
            # print("Creating classic detail generator.")
            self.D_detail = Generator(latent_dim=self.n_detail + self.n_detail_emo + self.n_cond, out_channels=1, out_scale=0.01,
                                      sample_mode='bilinear')
        elif str(self.config.detail_conditioning_type).lower() == "adain":
            # alex: not use for now
            # conditioning passed in through adain layers (this one is experimental and not currently used)
            # print("Creating AdaIn detail generator.")
            # self.D_detail = GeneratorAdaIn(self.n_detail + self.n_detail_emo,  self.n_cond, out_channels=1, out_scale=0.01,
            #                           sample_mode='bilinear')
            pass
        else:
            raise NotImplementedError(f"Detail conditioning invalid: '{self.config.detail_conditioning_type}'")

    def _create_model(self):
        # 1) build coarse encoder
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in self.config.keys():
            e_flame_type = self.config.e_flame_type

        if e_flame_type == 'ResnetEncoder':
            self.E_flame = ResnetEncoder(outsize=self.n_param)
        elif e_flame_type[:4] == 'swin':
            # alex: not use for now
            # self.E_flame = SwinEncoder(outsize=self.n_param, img_size=self.config.image_size, swin_type=e_flame_type)
            pass
        else:
            raise ValueError(f"Invalid 'e_flame_type' = {e_flame_type}")

        import copy 
        flame_cfg = copy.deepcopy(self.config)
        flame_cfg.n_shape = self._get_num_shape_params()
        if 'flame_mediapipe_lmk_embedding_path' not in flame_cfg.keys():
            # alex: not use for now
            # self.flame = FLAME(flame_cfg)
            pass
        else:
            self.flame = FLAME_mediapipe(flame_cfg)

        if self.uses_texture():
            # alex: not use for now
            # self.flametex = FLAMETex(self.config)
            pass
        else: 
            self.flametex = None

        # 2) build detail encoder
        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in self.config.keys():
            e_detail_type = self.config.e_detail_type

        if e_detail_type == 'ResnetEncoder':
            self.E_detail = ResnetEncoder(outsize=self.n_detail + self.n_detail_emo)
        elif e_flame_type[:4] == 'swin':
            # alex: not use for now
            # self.E_detail = SwinEncoder(outsize=self.n_detail + self.n_detail_emo, img_size=self.config.image_size, swin_type=e_detail_type)
            pass
        else:
            raise ValueError(f"Invalid 'e_detail_type'={e_detail_type}")
        self._create_detail_generator()
        # self._load_old_checkpoint()

    def _encode_flame(self, images):
        return self.E_flame(images)

    def decompose_code(self, code):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        code_list = []
        num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam,
                    self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list, None

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, detach=True):
        """
        Converts the displacement uv map (uv_z) and coarse_verts to a normal map coarse_normals. 
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)#.detach()
        if detach:
            uv_coarse_vertices = uv_coarse_vertices.detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals)#.detach()
        if detach:
            uv_coarse_normals = uv_coarse_normals.detach()

        uv_z = uv_z * self.uv_face_eye_mask

        # detail vertices = coarse vertice + predicted displacement*normals + fixed displacement*normals
        uv_detail_vertices = uv_coarse_vertices + \
                             uv_z * uv_coarse_normals + \
                             self.fixed_uv_dis[None, None, :,:] * uv_coarse_normals #.detach()

        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        # uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        # uv_detail_normals = util.gaussian_blur(uv_detail_normals)
        return uv_detail_normals, uv_coarse_vertices

    def visualize(self, visdict, savepath, catdim=1):
        grids = {}
        for key in visdict:
            # print(key)
            if visdict[key] is None:
                continue
            grids[key] = torchvision.utils.make_grid(
                F.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), catdim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath is not None:
            cv2.imwrite(savepath, grid_image)
        return grid_image

    def create_mesh(self, opdict, dense_template):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        if 'uv_texture_gt' in opdict.keys():
            texture = tensor2image(opdict['uv_texture_gt'][i])
        else:
            texture = None
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        if 'uv_detail_normals' in opdict.keys():
            normal_map = tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
            # upsample mesh, save detailed mesh
            texture = texture[:, :, [2, 1, 0]]
            normals = opdict['normals'][i].cpu().numpy()
            displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
            dense_vertices, dense_colors, dense_faces = upsample_mesh(vertices, normals, faces,
                                                                           displacement_map, texture, dense_template)
        else:
            normal_map = None
            dense_vertices = None
            dense_colors  = None
            dense_faces  = None

        return vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors

    def save_obj(self, filename, opdict, dense_template, mode ='detail'):
        if mode not in ['coarse', 'detail', 'both']:
            raise ValueError(f"Invalid mode '{mode}. Expected modes are: 'coarse', 'detail', 'both'")

        vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors \
            = self.create_mesh(opdict, dense_template)

        if mode == 'both':
            if isinstance(filename, list):
                filename_coarse = filename[0]
                filename_detail = filename[1]
            else:
                filename_coarse = filename
                filename_detail = filename.replace('.obj', '_detail.obj')
        elif mode == 'coarse':
            filename_coarse = filename
        else:
            filename_detail = filename

        if mode in ['coarse', 'both']:
            write_obj(str(filename_coarse), vertices, faces,
                            texture=texture,
                            uvcoords=uvcoords,
                            uvfaces=uvfaces,
                            normal_map=normal_map)

        if mode in ['detail', 'both']:
            write_obj(str(filename_detail),
                            dense_vertices,
                            dense_faces,
                            colors = dense_colors,
                            inverse_face_order=True)


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality. 
    Such as a separate expression decoder and related.
    """

    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        # E_flame should be fixed for expression EMOCA
        self.E_flame.requires_grad_(False)
        
        # 2) add expression decoder
        if self.config.expression_backbone == 'deca_parallel':
            # alex: not use for now
            ## a) Attach a parallel flow of FCs onto the original DECA coarse backbone. (Only the second FC head is trainable)
            # self.E_expression = SecondHeadResnet(self.E_flame, self.n_exp_param, 'same')
            pass
        elif self.config.expression_backbone == 'deca_clone':
            ## b) Clones the original DECA coarse decoder (and the entire decoder will be trainable) - This is in final EMOCA.
            #TODO: this will only work for Resnet. Make this work for the other backbones (Swin) as well.
            self.E_expression = ResnetEncoder(self.n_exp_param)
            # clone parameters of the ResNet
            self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        elif self.config.expression_backbone == 'emonet_trainable':
            # alex: not use for now
            # Trainable EmoNet instead of Resnet (deprecated)
            # self.E_expression = EmoNetRegressor(self.n_exp_param)
            pass
        elif self.config.expression_backbone == 'emonet_static':
            # alex: not use for now
            # Frozen EmoNet with a trainable head instead of Resnet (deprecated)
            # self.E_expression = EmonetRegressorStatic(self.n_exp_param)
            pass
        else:
            raise ValueError(f"Invalid expression backbone: '{self.config.expression_backbone}'")
        
        if self.config.get('zero_out_last_enc_layer', False):
            self.E_expression.reset_last_layer() 

    def _reconfigure(self, config):
        super()._reconfigure(config)
        self.n_exp_param = self.config.n_exp

        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            self.n_exp_param += self.config.n_pose
        elif self.config.exp_deca_global_pose or self.config.exp_deca_jaw_pose:
            self.n_exp_param += 3

    def _encode_flame(self, images):
        if self.config.expression_backbone == 'deca_parallel':
            #SecondHeadResnet does the forward pass for shape and expression at the same time
            return self.E_expression(images)
        # other regressors have to do a separate pass over the image
        deca_code = super()._encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]

        deca_code_list, _ = super().decompose_code(deca_code)
        # shapecode, texcode, expcode, posecode, cam, lightcode = deca_code_list
        exp_idx = 2
        pose_idx = 3

        # deca_exp_code = deca_code_list[exp_idx]
        # deca_global_pose_code = deca_code_list[pose_idx][:3]
        # deca_jaw_pose_code = deca_code_list[pose_idx][3:6]

        deca_code_list_copy = deca_code_list.copy()

        # self.E_mica.cfg.model.n_shape

        #TODO: clean this if-else block up
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            exp_code = expdeca_code[:, :self.config.n_exp]
            pose_code = expdeca_code[:, self.config.n_exp:]
            deca_code_list[exp_idx] = exp_code
            deca_code_list[pose_idx] = pose_code
        elif self.config.exp_deca_global_pose:
            # global pose from ExpDeca, jaw pose from EMOCA
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_exp_deca, pose_code_deca[:,3:]], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        elif self.config.exp_deca_jaw_pose:
            # global pose from EMOCA, jaw pose from ExpDeca
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_deca[:, :3], pose_code_exp_deca], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        else:
            exp_code = expdeca_code
            deca_code_list[exp_idx] = exp_code

        return deca_code_list, deca_code_list_copy


class DecaModule(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks. 
    """

    def __init__(self, model_params, learning_params, inout_params, stage_name = ""):
        """
        :param model_params: a DictConfig of parameters about the model itself
        :param learning_params: a DictConfig of parameters corresponding to the learning process (such as optimizer, lr and others)
        :param inout_params: a DictConfig of parameters about input and output (where checkpoints and visualizations are saved)
        """
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params

        # detail conditioning - what is given as the conditioning input to the detail generator in detail stage training
        if 'detail_conditioning' not in model_params.keys():
            # jaw, expression and detail code by default
            self.detail_conditioning = ['jawpose', 'expression', 'detail'] 
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detail_conditioning = self.detail_conditioning
        else:
            self.detail_conditioning = model_params.detail_conditioning

        # deprecated and is not used
        if 'detailemo_conditioning' not in model_params.keys():
            self.detailemo_conditioning = []
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detailemo_conditioning = self.detailemo_conditioning
        else:
            self.detailemo_conditioning = model_params.detailemo_conditioning

        supported_conditioning_keys = ['identity', 'jawpose', 'expression', 'detail', 'detailemo']
        
        for c in self.detail_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")
        for c in self.detailemo_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")

        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)

        self.mode = 1 if str(model_params.mode).upper() == "COARSE" else 2
        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        
        # initialize the emotion perceptual loss (used for EMOCA supervision)
        self.emonet_loss = None
        # self._init_emotion_loss()
        
        # initialize the au perceptual loss (not currently used in EMOCA)
        self.au_loss = None
        # self._init_au_loss()

        # initialize the lip reading perceptual loss (not currently used in original EMOCA)
        self.lipread_loss = None
        # self._init_lipread_loss()

        # MPL regressor from the encoded space to emotion labels (not used in EMOCA but could be used for direct emotion supervision)
        if 'mlp_emotion_predictor' in self.deca.config.keys():
            # alex: not use for now
            # self._build_emotion_mlp(self.deca.config.mlp_emotion_predictor)
            # self.emotion_mlp = EmotionMLP(self.deca.config.mlp_emotion_predictor, model_params)
            pass
        else:
            self.emotion_mlp = None

    def get_input_image_size(self): 
        return (self.deca.config.image_size, self.deca.config.image_size)

    def _instantiate_deca(self, model_params):
        """
        Instantiate the DECA network.
        """
        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)

    def uses_texture(self):
        """
        Check if the model uses texture
        """
        return self.deca.uses_texture()

    def visualize(self, visdict, savepath, catdim=1):
        return self.deca.visualize(visdict, savepath, catdim)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        return self

    def cpu(self):
        super().cpu()
        return self

    def forward(self, batch):
        values = self.encode(batch, training=False)
        values = self.decode(values, training=False)
        return values

    def _unwrap_list(self, codelist): 
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return shapecode, texcode, expcode, posecode, cam, lightcode

    def _unwrap_list_to_dict(self, codelist): 
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return {'shape': shapecode, 'tex': texcode, 'exp': expcode, 'pose': posecode, 'cam': cam, 'light': lightcode}
        # return shapecode, texcode, expcode, posecode, cam, lightcode

    def _encode_flame(self, images):
        if self.mode == 1 or \
                (self.mode == 2 and self.deca.config.train_coarse):
            # forward pass with gradients (for coarse stage (used), or detail stage with coarse training (not used))
            parameters = self.deca._encode_flame(images)
        elif self.mode == 2:
            # in detail stage, the coarse forward pass does not need gradients
            with torch.no_grad():
                parameters = self.deca._encode_flame(images)
        else:
            raise ValueError(f"Invalid EMOCA Mode {self.mode}")
        code_list, original_code = self.deca.decompose_code(parameters)
        return code_list, original_code

    def encode(self, batch, training=True) -> dict:
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        For a testing pass, the images suffice. 
        :param training: Whether the forward pass is for training or testing.
        """
        codedict = {}
        original_batch_size = batch['image'].shape[0]

        images = batch['image']

        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        if 'landmark' in batch.keys():
            lmk = batch['landmark']
            lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        
        if 'landmark_mediapipe' in batch.keys():
            lmk_mp = batch['landmark_mediapipe']
            lmk_mp = lmk_mp.view(-1, lmk_mp.shape[-2], lmk_mp.shape[-1])
        else:
            lmk_mp = None

        if 'mask' in batch.keys():
            masks = batch['mask']
            masks = masks.view(-1, images.shape[-2], images.shape[-1])

        # valence / arousal - not necessary unless we want to use VA for supervision (not done in EMOCA)
        if 'va' in batch:
            va = batch['va']
            va = va.view(-1, va.shape[-1])
        else:
            va = None

        # 7 basic expression - not necessary unless we want to use expression for supervision (not done in EMOCA or DECA)
        if 'expr7' in batch:
            expr7 = batch['expr7']
            expr7 = expr7.view(-1, expr7.shape[-1])
        else:
            expr7 = None

        # affectnet basic expression - not necessary unless we want to use expression for supervision (not done in EMOCA or DECA)
        if 'affectnetexp' in batch:
            affectnetexp = batch['affectnetexp']
            affectnetexp = affectnetexp.view(-1, affectnetexp.shape[-1])
        else:
            affectnetexp = None

        # expression weights if supervising by expression is used (to balance the classification loss) - not done in EMOCA or DECA
        if 'expression_weight' in batch:
            exprw = batch['expression_weight']
            exprw = exprw.view(-1, exprw.shape[-1])
        else:
            exprw = None


        # 1) COARSE STAGE
        # forward pass of the coarse encoder
        # shapecode, texcode, expcode, posecode, cam, lightcode = self._encode_flame(images)
        code, original_code = self._encode_flame(images)
        shapecode, texcode, expcode, posecode, cam, lightcode = self._unwrap_list(code)
        if original_code is not None:
            original_code = self._unwrap_list_to_dict(original_code)

        if training:
            # If training, we employ the disentanglement strategy
            if self.mode == 1:

                if self.deca.config.shape_constrain_type == 'same':
                    ## Enforce that all identity shape codes within ring are the same. The batch is duplicated 
                    ## and the duplicated part's shape codes are shuffled.

                    # reshape shapecode => [B, K, n_shape]
                    # shapecode_idK = shapecode.view(self.batch_size, self.deca.K, -1)
                    shapecode_idK = shapecode.view(original_batch_size, K, -1)
                    # get mean id
                    shapecode_mean = torch.mean(shapecode_idK, dim=[1])
                    # shapecode_new = shapecode_mean[:, None, :].repeat(1, self.deca.K, 1)
                    shapecode_new = shapecode_mean[:, None, :].repeat(1, K, 1)
                    shapecode = shapecode_new.view(-1, self.deca._get_num_shape_params())

                    # do the same for the original code dict
                    shapecode_orig = original_code['shape']
                    shapecode_orig_idK = shapecode_orig.view(original_batch_size, K, -1)
                    shapecode_orig_mean = torch.mean(shapecode_orig_idK, dim=[1])
                    shapecode_orig_new = shapecode_orig_mean[:, None, :].repeat(1, K, 1)
                    original_code['shape'] = shapecode_orig_new.view(-1, self.deca._get_num_shape_params())

                elif self.deca.config.shape_constrain_type == 'exchange':
                    ## Shuffle identitys shape codes within ring (they should correspond to the same identity)
                    '''
                    make sure s0, s1 is something to make shape close
                    the difference from ||so - s1|| is 
                    the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                    '''
                    # new_order = np.array([np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(self.deca.config.batch_size_train)])
                    # new_order = np.array([np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(original_batch_size)])
                    new_order = np.array([np.random.permutation(K) + i * K for i in range(original_batch_size)])
                    new_order = new_order.flatten()
                    shapecode_new = shapecode[new_order]
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode_new], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    if lmk_mp is not None:
                        lmk_mp = torch.cat([lmk_mp, lmk_mp], dim=0)
                    masks = torch.cat([masks, masks], dim=0)

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                    # do the same for the original code dict
                    shapecode_orig = original_code['shape']
                    shapecode_orig_new = shapecode_orig[new_order]
                    original_code['shape'] = torch.cat([shapecode_orig, shapecode_orig_new], dim=0)
                    original_code['tex'] = torch.cat([original_code['tex'], original_code['tex']], dim=0)
                    original_code['exp'] = torch.cat([original_code['exp'], original_code['exp']], dim=0)
                    original_code['pose'] = torch.cat([original_code['pose'], original_code['pose']], dim=0)
                    original_code['cam'] = torch.cat([original_code['cam'], original_code['cam']], dim=0)
                    original_code['light'] = torch.cat([original_code['light'], original_code['light']], dim=0)


                elif self.deca.config.shape_constrain_type == 'shuffle_expression':
                    assert original_code is not None
                    ## DEPRECATED, NOT USED IN EMOCA OR DECA
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    # exchange expression
                    expcode_new = expcode[new_order]
                    expcode = torch.cat([expcode, expcode_new], dim=0)

                    # exchange jaw pose (but not global pose)
                    global_pose = posecode[:, :3]
                    jaw_pose = posecode[:, 3:]
                    jaw_pose_new = jaw_pose[new_order]
                    jaw_pose = torch.cat([jaw_pose, jaw_pose_new], dim=0)
                    global_pose = torch.cat([global_pose, global_pose], dim=0)
                    posecode = torch.cat([global_pose, jaw_pose], dim=1)

                    ## duplicate the rest
                    shapecode = torch.cat([shapecode, shapecode], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## duplicate gt if any
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    print(f"TRAINING: {training}")
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    if lmk_mp is not None:
                        lmk_mp = torch.cat([lmk_mp, lmk_mp], dim=0)
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, old_order])
                    ref_images_expression_idxs = np.concatenate([old_order, new_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va[new_order]], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7[new_order]], dim=0)

                    # do the same for the original code dict
                    original_code['shape'] = torch.cat([original_code['shape'], original_code['shape']], dim=0)
                    original_code['tex'] = torch.cat([original_code['tex'], original_code['tex']], dim=0)
                    original_code['exp'] = torch.cat([original_code['exp'], original_code['exp'][new_order]], dim=0)
                    original_global_pose = original_code['pose'][:, :3]
                    original_jaw_pose = original_code['pose'][:, 3:]
                    original_jaw_pose = torch.cat([original_jaw_pose, original_jaw_pose[new_order]], dim=0)
                    original_global_pose = torch.cat([original_global_pose, original_global_pose], dim=0)
                    original_code['pose'] = torch.cat([original_global_pose, original_jaw_pose], dim=1)
                    original_code['cam'] = torch.cat([original_code['cam'], original_code['cam']], dim=0)
                    original_code['light'] = torch.cat([original_code['light'], original_code['light']], dim=0)

                elif self.deca.config.shape_constrain_type == 'shuffle_shape':
                    ## The shape codes are shuffled without duplication
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    shapecode_new = shapecode[new_order]
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode_new], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, new_order])
                    ref_images_expression_idxs = np.concatenate([old_order, old_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                    # do the same for the original code dict
                    shapecode_orig = original_code['shape']
                    shapecode_orig_new = shapecode_orig[new_order]
                    original_code['shape'] = torch.cat([shapecode_orig, shapecode_orig_new], dim=0)
                    original_code['tex'] = torch.cat([original_code['tex'], original_code['tex']], dim=0)
                    original_code['exp'] = torch.cat([original_code['exp'], original_code['exp']], dim=0)
                    original_code['pose'] = torch.cat([original_code['pose'], original_code['pose']], dim=0)
                    original_code['cam'] = torch.cat([original_code['cam'], original_code['cam']], dim=0)
                    original_code['light'] = torch.cat([original_code['light'], original_code['light']], dim=0)
                    original_code['ref_images_identity_idxs'] = ref_images_identity_idxs
                    original_code['ref_images_expression_idxs'] = ref_images_expression_idxs

                elif 'expression_constrain_type' in self.deca.config.keys() and \
                        self.deca.config.expression_constrain_type == 'same':
                    ## NOT USED IN EMOCA OR DECA, deprecated

                    # reshape shapecode => [B, K, n_shape]
                    # shapecode_idK = shapecode.view(self.batch_size, self.deca.K, -1)
                    expcode_idK = expcode.view(original_batch_size, K, -1)
                    # get mean id
                    expcode_mean = torch.mean(expcode_idK, dim=[1])
                    # shapecode_new = shapecode_mean[:, None, :].repeat(1, self.deca.K, 1)
                    expcode_new = expcode_mean[:, None, :].repeat(1, K, 1)
                    expcode = expcode_new.view(-1, self.deca._get_num_shape_params())

                    # do the same thing for the original code dict
                    expcode_idK = original_code['exp'].view(original_batch_size, K, -1)
                    expcode_mean = torch.mean(expcode_idK, dim=[1])
                    expcode_new = expcode_mean[:, None, :].repeat(1, K, 1)
                    original_code['exp'] = expcode_new.view(-1, self.deca._get_num_shape_params())

                elif 'expression_constrain_type' in self.deca.config.keys() and \
                        self.deca.config.expression_constrain_type == 'exchange':
                    ## NOT USED IN EMOCA OR DECA, deprecated
                    expcode, posecode, shapecode, lightcode, texcode, images, cam, lmk, masks, va, expr7, affectnetexp, _, _, exprw, lmk_mp = \
                        self._expression_ring_exchange(original_batch_size, K,
                                  expcode, posecode, shapecode, lightcode, texcode,
                                  images, cam, lmk, masks, va, expr7, affectnetexp, None, None, exprw, lmk_mp)
                    # (self, original_batch_size, K,
                    #                                   expcode, posecode, shapecode, lightcode, texcode,
                    #                                   images, cam, lmk, masks, va, expr7, affectnetexp,
                    #                                   detailcode=None, detailemocode=None, exprw=None):

        # 2) DETAIL STAGE
        if self.mode == 2:
            all_detailcode = self.deca.E_detail(images)

            # identity-based detail code
            detailcode = all_detailcode[:, :self.deca.n_detail]

            # detail emotion code is deprecated and will be empty
            detailemocode = all_detailcode[:, self.deca.n_detail:(self.deca.n_detail + self.deca.n_detail_emo)]

            if training:
                # If training, we employ the disentanglement strategy
                if self.deca.config.detail_constrain_type == 'exchange':
                    # Identity within the same ring should be the same, so they should have the same code. 
                    # This can be enforced by shuffling. The batch is duplicated and the duplicated part's code shuffled
                    '''
                    make sure s0, s1 is something to make shape close
                    the difference from ||so - s1|| is 
                    the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                    '''
                    # this creates a per-ring random permutation. The detail exchange happens ONLY between the same
                    # identities (within the ring) but not outside (no cross-identity detail exchange)
                    new_order = np.array(
                        # [np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(original_batch_size)])
                        [np.random.permutation(K) + i * K for i in range(original_batch_size)])
                    new_order = new_order.flatten()
                    detailcode_new = detailcode[new_order]
                    detailcode = torch.cat([detailcode, detailcode_new], dim=0)
                    detailemocode = torch.cat([detailemocode, detailemocode], dim=0)
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                elif self.deca.config.detail_constrain_type == 'shuffle_expression':
                    ## Deprecated and not used in EMOCA or DECA
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    # exchange expression
                    expcode_new = expcode[new_order]
                    expcode = torch.cat([expcode, expcode_new], dim=0)

                    # exchange emotion code, but not (identity-based) detailcode
                    detailemocode_new = detailemocode[new_order]
                    detailemocode = torch.cat([detailemocode, detailemocode_new], dim=0)
                    detailcode = torch.cat([detailcode, detailcode], dim=0)

                    # exchange jaw pose (but not global pose)
                    global_pose = posecode[:, :3]
                    jaw_pose = posecode[:, 3:]
                    jaw_pose_new = jaw_pose[new_order]
                    jaw_pose = torch.cat([jaw_pose, jaw_pose_new], dim=0)
                    global_pose = torch.cat([global_pose, global_pose], dim=0)
                    posecode = torch.cat([global_pose, jaw_pose], dim=1)


                    ## duplicate the rest
                    shapecode = torch.cat([shapecode, shapecode], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## duplicate gt if any
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    print(f"TRAINING: {training}")
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, old_order])
                    ref_images_expression_idxs = np.concatenate([old_order, new_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va[new_order]], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7[new_order]], dim=0)

                elif self.deca.config.detail_constrain_type == 'shuffle_shape':
                    ## Shuffles teh shape code without duplicating the batch
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    shapecode_new = shapecode[new_order]
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode_new], dim=0)

                    # exchange (identity-based) detailcode, but not emotion code
                    detailcode_new = detailcode[new_order]
                    detailcode = torch.cat([detailcode, detailcode_new], dim=0)
                    detailemocode = torch.cat([detailemocode, detailemocode], dim=0)

                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, new_order])
                    ref_images_expression_idxs = np.concatenate([old_order, old_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                elif 'expression_constrain_type' in self.deca.config.keys() and \
                        self.deca.config.expression_constrain_type == 'exchange':
                    expcode, posecode, shapecode, lightcode, texcode, images, cam, lmk, masks, va, expr7, affectnetexp, detailcode, detailemocode, exprw = \
                        self._expression_ring_exchange(original_batch_size, K,
                                  expcode, posecode, shapecode, lightcode, texcode,
                                  images, cam, lmk, masks, va, expr7, affectnetexp, detailcode, detailemocode, exprw)


        codedict['shapecode'] = shapecode
        codedict['texcode'] = texcode
        codedict['expcode'] = expcode
        codedict['posecode'] = posecode
        codedict['cam'] = cam
        codedict['lightcode'] = lightcode
        if self.mode == 2:
            codedict['detailcode'] = detailcode
            codedict['detailemocode'] = detailemocode
        codedict['images'] = images
        if 'mask' in batch.keys():
            codedict['masks'] = masks
        if 'landmark' in batch.keys():
            codedict['lmk'] = lmk
        if lmk_mp is not None:
            codedict['lmk_mp'] = lmk_mp

        if 'va' in batch.keys():
            codedict['va'] = va
        if 'expr7' in batch.keys():
            codedict['expr7'] = expr7
        if 'affectnetexp' in batch.keys():
            codedict['affectnetexp'] = affectnetexp

        if 'expression_weight' in batch.keys():
            codedict['expression_weight'] = exprw

        if original_code is not None:
            codedict['original_code'] = original_code

        return codedict

    def _create_conditioning_lists(self, codedict, condition_list):
        detail_conditioning_list = []
        if 'globalpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, :3]]
        if 'jawpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, 3:]]
        if 'identity' in condition_list:
            detail_conditioning_list += [codedict["shapecode"]]
        if 'expression' in condition_list:
            detail_conditioning_list += [codedict["expcode"]]

        if isinstance(self.deca.D_detail, Generator):
            # the detail codes might be excluded from conditioning based on the Generator architecture (for instance
            # for AdaIn Generator)
            if 'detail' in condition_list:
                detail_conditioning_list += [codedict["detailcode"]]
            if 'detailemo' in condition_list:
                detail_conditioning_list += [codedict["detailemocode"]]

        return detail_conditioning_list

    def decode(self, codedict, training=True, render=True, **kwargs) -> dict:
        """
        Forward decoding pass of the model. Takes the latent code predicted by the encoding stage and reconstructs and renders the shape.
        :param codedict: Batch dict of the predicted latent codes
        :param training: Whether the forward pass is for training or testing.
        """
        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict['images']
        if 'masks' in codedict.keys():
            masks = codedict['masks']
        else:
            masks = None

        effective_batch_size = images.shape[0]  # this is the current batch size after all training augmentations modifications

        # 1) Reconstruct the face mesh
        # FLAME - world space
        if not isinstance(self.deca.flame, FLAME_mediapipe):
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                          pose_params=posecode)
            landmarks2d_mediapipe = None
        else:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)
        # world to camera
        trans_verts = batch_orth_proj(verts, cam)
        predicted_landmarks = batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        if landmarks2d_mediapipe is not None:
            predicted_landmarks_mediapipe = batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]
            predicted_landmarks_mediapipe[:, :, 1:] = - predicted_landmarks_mediapipe[:, :, 1:]

        if self.uses_texture():
            albedo = self.deca.flametex(texcode)
        else: 
            # if not using texture, default to gray
            albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, self.deca.config.uv_size], device=images.device) * 0.5

        # 2) Render the coarse image
        if render:
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                        ops['grid'].detach(),
                                        align_corners=False)
            # images
            predicted_images = ops['images']
            # predicted_images = ops['images'] * mask_face_eye * ops['alpha_images']
            # predicted_images_no_mask = ops['images'] #* mask_face_eye * ops['alpha_images']
            segmentation_type = None
            if isinstance(self.deca.config.useSeg, bool):
                if self.deca.config.useSeg:
                    segmentation_type = 'gt'
                else:
                    segmentation_type = 'rend'
            elif isinstance(self.deca.config.useSeg, str):
                segmentation_type = self.deca.config.useSeg
            else:
                raise RuntimeError(f"Invalid 'useSeg' type: '{type(self.deca.config.useSeg)}'")

            if segmentation_type not in ["gt", "rend", "intersection", "union"]:
                raise ValueError(f"Invalid segmentation type for masking '{segmentation_type}'")

            if masks is None: # if mask not provided, the only mask available is the rendered one
                segmentation_type = 'rend'

            elif masks.shape[-1] != predicted_images.shape[-1] or masks.shape[-2] != predicted_images.shape[-2]:
                # resize masks if need be (this is only done if configuration was changed at some point after training)
                dims = masks.ndim == 3
                if dims:
                    masks = masks[:, None, :, :]
                masks = F.interpolate(masks, size=predicted_images.shape[-2:], mode='bilinear')
                if dims:
                    masks = masks[:, 0, ...]

            # resize images if need be (this is only done if configuration was changed at some point after training)
            if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                ## special case only for inference time if the rendering image sizes have been changed
                images_resized = F.interpolate(images, size=predicted_images.shape[-2:], mode='bilinear')
            else:
                images_resized = images

            # what type of segmentation we use
            if segmentation_type == "gt": # GT stands for external segmetnation predicted by face parsing or similar
                masks = masks[:, None, :, :]
            elif segmentation_type == "rend": # mask rendered as a silhouette of the face mesh
                masks = mask_face_eye * ops['alpha_images']
            elif segmentation_type == "intersection": # intersection of the two above
                masks = masks[:, None, :, :] * mask_face_eye * ops['alpha_images']
            elif segmentation_type == "union": # union of the first two options
                masks = torch.max(masks[:, None, :, :],  mask_face_eye * ops['alpha_images'])
            else:
                raise RuntimeError(f"Invalid segmentation type for masking '{segmentation_type}'")


            if self.deca.config.background_from_input in [True, "input"]:
                if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                    ## special case only for inference time if the rendering image sizes have been changed
                    predicted_images = (1. - masks) * images_resized + masks * predicted_images
                else:
                    predicted_images = (1. - masks) * images + masks * predicted_images
            elif self.deca.config.background_from_input in [False, "black"]:
                predicted_images = masks * predicted_images
            elif self.deca.config.background_from_input in ["none"]:
                predicted_images = predicted_images
            else:
                raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")

        # 3) Render the detail image
        if self.mode == 2:
            detailcode = codedict['detailcode']
            detailemocode = codedict['detailemocode']

            # a) Create the detail conditioning lists
            detail_conditioning_list = self._create_conditioning_lists(codedict, self.detail_conditioning)
            detailemo_conditioning_list = self._create_conditioning_lists(codedict, self.detailemo_conditioning)
            final_detail_conditioning_list = detail_conditioning_list + detailemo_conditioning_list


            # b) Pass the detail code and the conditions through the detail generator to get displacement UV map
            if isinstance(self.deca.D_detail, Generator):
                uv_z = self.deca.D_detail(torch.cat(final_detail_conditioning_list, dim=1))
            # alex: not use for now
            # elif isinstance(self.deca.D_detail, GeneratorAdaIn):
            #     uv_z = self.deca.D_detail(z=torch.cat([detailcode, detailemocode], dim=1),
            #                               cond=torch.cat(final_detail_conditioning_list, dim=1))
            else:
                raise ValueError(f"This class of generarator is not supported: '{self.deca.D_detail.__class__.__name__}'")

            # if there is a displacement mask, apply it (DEPRECATED and not USED in DECA or EMOCA)
            if hasattr(self.deca, 'displacement_mask') and self.deca.displacement_mask is not None:
                if 'apply_displacement_masks' in self.deca.config.keys() and self.deca.config.apply_displacement_masks:
                    uv_z = uv_z * self.deca.displacement_mask

            # uv_z = self.deca.D_detail(torch.cat([posecode[:, 3:], expcode, detailcode], dim=1))
            # render detail
            if render:
                detach_from_coarse_geometry = not self.deca.config.train_coarse
                uv_detail_normals, uv_coarse_vertices = self.deca.displacement2normal(uv_z, verts, ops['normals'],
                                                                                    detach=detach_from_coarse_geometry)
                uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
                uv_texture = albedo.detach() * uv_shading

                # batch size X image_rows X image_cols X 2
                # you can query the grid for UV values of the face mesh at pixel locations
                grid = ops['grid']
                if detach_from_coarse_geometry:
                    # if the grid is detached, the gradient of the positions of UV-values in image space won't flow back to the geometry
                    grid = grid.detach()
                predicted_detailed_image = F.grid_sample(uv_texture, grid, align_corners=False)
                if self.deca.config.background_from_input in [True, "input"]:
                    if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                        ## special case only for inference time if the rendering image sizes have been changed
                        # images_resized = F.interpolate(images, size=predicted_images.shape[-2:], mode='bilinear')
                        ## before bugfix
                        # predicted_images = (1. - masks) * images_resized + masks * predicted_images
                        ## after bugfix
                        predicted_detailed_image = (1. - masks) * images_resized + masks * predicted_detailed_image
                    else:
                        predicted_detailed_image = (1. - masks) * images + masks * predicted_detailed_image
                elif self.deca.config.background_from_input in [False, "black"]:
                    predicted_detailed_image = masks * predicted_detailed_image
                elif self.deca.config.background_from_input in ["none"]:
                    predicted_detailed_image = predicted_detailed_image
                else:
                    raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")


                # --- extract texture
                uv_pverts = self.deca.render.world2uv(trans_verts).detach()
                uv_gt = F.grid_sample(torch.cat([images_resized, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                                    mode='bilinear')
                uv_texture_gt = uv_gt[:, :3, :, :].detach()
                uv_mask_gt = uv_gt[:, 3:, :, :].detach()
                # self-occlusion
                normals = vertex_normals(trans_verts, self.deca.render.faces.expand(effective_batch_size, -1, -1))
                uv_pnorm = self.deca.render.world2uv(normals)

                uv_mask = (uv_pnorm[:, -1, :, :] < -0.05).float().detach()
                uv_mask = uv_mask[:, None, :, :]
                ## combine masks
                uv_vis_mask = uv_mask_gt * uv_mask * self.deca.uv_face_eye_mask
        else:
            uv_detail_normals = None
            predicted_detailed_image = None


        ## 4) (Optional) NEURAL RENDERING - not used in neither DECA nor EMOCA
        # If neural rendering is enabled, the differentiable rendered synthetic images are translated using an image translation net (such as StarGan)
        predicted_translated_image = None
        predicted_detailed_translated_image = None
        translated_uv_texture = None

        if render:
            if self.deca._has_neural_rendering():
                predicted_translated_image = self.deca.image_translator(
                    {
                        "input_image" : predicted_images,
                        "ref_image" : images,
                        "target_domain" : torch.tensor([0]*predicted_images.shape[0],
                                                    dtype=torch.int64, device=predicted_images.device)
                    }
                )

                if self.mode == 2:
                    predicted_detailed_translated_image = self.deca.image_translator(
                            {
                                "input_image" : predicted_detailed_image,
                                "ref_image" : images,
                                "target_domain" : torch.tensor([0]*predicted_detailed_image.shape[0],
                                                            dtype=torch.int64, device=predicted_detailed_image.device)
                            }
                        )
                    translated_uv = F.grid_sample(torch.cat([predicted_detailed_translated_image, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                                        mode='bilinear')
                    translated_uv_texture = translated_uv[:, :3, :, :].detach()

                else:
                    predicted_detailed_translated_image = None

                    translated_uv_texture = None
                    # no need in coarse mode
                    # translated_uv = F.grid_sample(torch.cat([predicted_translated_image, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                    #                       mode='bilinear')
                    # translated_uv_texture = translated_uv_gt[:, :3, :, :].detach()

        if self.emotion_mlp is not None:
            codedict = self.emotion_mlp(codedict, "emo_mlp_")

        # populate the value dict for metric computation/visualization
        if render:
            codedict['predicted_images'] = predicted_images
            codedict['predicted_detailed_image'] = predicted_detailed_image
            codedict['predicted_translated_image'] = predicted_translated_image
            codedict['ops'] = ops
            codedict['normals'] = ops['normals']
            codedict['mask_face_eye'] = mask_face_eye
        
        codedict['verts'] = verts
        codedict['albedo'] = albedo
        codedict['landmarks2d'] = landmarks2d
        codedict['landmarks3d'] = landmarks3d
        codedict['predicted_landmarks'] = predicted_landmarks
        if landmarks2d_mediapipe is not None:
            codedict['predicted_landmarks_mediapipe'] = predicted_landmarks_mediapipe
        codedict['trans_verts'] = trans_verts
        codedict['masks'] = masks

        if self.mode == 2:
            if render:
                codedict['predicted_detailed_translated_image'] = predicted_detailed_translated_image
                codedict['translated_uv_texture'] = translated_uv_texture
                codedict['uv_texture_gt'] = uv_texture_gt
                codedict['uv_texture'] = uv_texture
                codedict['uv_detail_normals'] = uv_detail_normals
                codedict['uv_shading'] = uv_shading
                codedict['uv_vis_mask'] = uv_vis_mask
                codedict['uv_mask'] = uv_mask
            codedict['uv_z'] = uv_z
            codedict['displacement_map'] = uv_z + self.deca.fixed_uv_dis[None, None, :, :]

        return codedict

    def _cut_mouth_vectorized(self, images, landmarks, convert_grayscale=True):
        # mouth_window_margin = 12
        mouth_window_margin = 1 # not temporal
        mouth_crop_height = 96
        mouth_crop_width = 96
        mouth_landmark_start_idx = 48
        mouth_landmark_stop_idx = 68
        B, T = images.shape[:2]

        landmarks = landmarks.to(torch.float32)

        with torch.no_grad():
            image_size = images.shape[-1] / 2

            landmarks = landmarks * image_size + image_size
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2) 
            # reshape to (T, 136) 
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1)
            # make temporal dimension last 
            landmarks_t = landmarks_t.permute(0, 2, 1)
            # change chape to (N, 136, T)
            # landmarks_t = landmarks_t.unsqueeze(0)
            # smooth with temporal convolution
            temporal_filter = torch.ones(mouth_window_margin, device=images.device) / mouth_window_margin
            # pad the the landmarks 
            landmarks_t_padded = F.pad(landmarks_t, (mouth_window_margin // 2, mouth_window_margin // 2), mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            if temporal_filter.numel() > 1:
                smooth_landmarks_t = F.conv1d(landmarks_t_padded, 
                    temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels,1,temporal_filter.numel()), 
                    groups=num_channels, padding='valid'
                )
                smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]
            else:
                smooth_landmarks_t = landmarks_t

            # reshape back to the original shape 
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., mouth_landmark_start_idx:mouth_landmark_stop_idx, :]
            
            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
        
            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image 
            # create the grid
            height = mouth_crop_height//2
            width = mouth_crop_width//2

            torch.arange(0, mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, mouth_crop_height).to(images.device) / (images.shape[-2] /2),
                                            torch.linspace(-width, width, mouth_crop_width).to(images.device) / (images.shape[-1] /2) ), 
                                            dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)

            center_x_t -= images.shape[-1] / 2
            center_y_t -= images.shape[-2] / 2

            center_x_t /= images.shape[-1] / 2
            center_y_t /= images.shape[-2] / 2

            grid = grid + torch.cat([center_x_t, center_y_t ], dim=-1).unsqueeze(-2).unsqueeze(-2)

        images = images.view(B*T, *images.shape[2:])
        grid = grid.view(B*T, *grid.shape[2:])

        if convert_grayscale: 
            images = F_v.rgb_to_grayscale(images)

        image_crops = F.grid_sample(
            images, 
            grid,  
            align_corners=True, 
            padding_mode='zeros',
            mode='bicubic'
            )
        image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        if convert_grayscale:
            image_crops = image_crops

        return image_crops

    def _metric_or_loss(self, loss_dict, metric_dict, is_loss):
        if is_loss:
            d = loss_dict
        else:
            d = metric_dict
        return d

    def _val_to_be_logged(self, d):
        if not hasattr(self, 'val_dict_list'):
            self.val_dict_list = []
        self.val_dict_list += [d]

    def _get_logging_prefix(self):
        prefix = self.stage_name + str(self.mode.name).lower()
        return prefix

    def _visualization_checkpoint(self, verts, trans_verts, ops, uv_detail_normals, additional, batch_idx, stage, prefix,
                                  save=False):
        batch_size = verts.shape[0]
        visind = np.arange(batch_size)
        shape_images = self.deca.render.render_shape(verts, trans_verts)
        if uv_detail_normals is not None:
            detail_normal_images = F.grid_sample(uv_detail_normals.detach(), ops['grid'].detach(),
                                                 align_corners=False)
            shape_detail_images = self.deca.render.render_shape(verts, trans_verts,
                                                           detail_normal_images=detail_normal_images)
        else:
            shape_detail_images = None

        visdict = {}
        if 'images' in additional.keys():
            visdict['inputs'] = additional['images'][visind]

        if 'images' in additional.keys() and 'lmk' in additional.keys():
            visdict['landmarks_gt'] = tensor_vis_landmarks(additional['images'][visind], additional['lmk'][visind])

        if 'images' in additional.keys() and 'predicted_landmarks' in additional.keys():
            visdict['landmarks_predicted'] = tensor_vis_landmarks(additional['images'][visind],
                                                                     additional['predicted_landmarks'][visind])

        if 'predicted_images' in additional.keys():
            visdict['output_images_coarse'] = additional['predicted_images'][visind]

        if 'predicted_translated_image' in additional.keys() and additional['predicted_translated_image'] is not None:
            visdict['output_translated_images_coarse'] = additional['predicted_translated_image'][visind]

        visdict['geometry_coarse'] = shape_images[visind]
        if shape_detail_images is not None:
            visdict['geometry_detail'] = shape_detail_images[visind]

        if 'albedo_images' in additional.keys():
            visdict['albedo_images'] = additional['albedo_images'][visind]

        if 'masks' in additional.keys():
            visdict['mask'] = additional['masks'].repeat(1, 3, 1, 1)[visind]
        if 'albedo' in additional.keys():
            visdict['albedo'] = additional['albedo'][visind]

        if 'predicted_detailed_image' in additional.keys() and additional['predicted_detailed_image'] is not None:
            visdict['output_images_detail'] = additional['predicted_detailed_image'][visind]

        if 'predicted_detailed_translated_image' in additional.keys() and additional['predicted_detailed_translated_image'] is not None:
            visdict['output_translated_images_detail'] = additional['predicted_detailed_translated_image'][visind]

        if 'shape_detail_images' in additional.keys():
            visdict['shape_detail_images'] = additional['shape_detail_images'][visind]

        if 'uv_detail_normals' in additional.keys():
            visdict['uv_detail_normals'] = additional['uv_detail_normals'][visind] * 0.5 + 0.5

        if 'uv_texture_patch' in additional.keys():
            visdict['uv_texture_patch'] = additional['uv_texture_patch'][visind]

        if 'uv_texture_gt' in additional.keys():
            visdict['uv_texture_gt'] = additional['uv_texture_gt'][visind]

        if 'translated_uv_texture' in additional.keys() and additional['translated_uv_texture'] is not None:
            visdict['translated_uv_texture'] = additional['translated_uv_texture'][visind]

        if 'uv_vis_mask_patch' in additional.keys():
            visdict['uv_vis_mask_patch'] = additional['uv_vis_mask_patch'][visind]

        if save:
            savepath = f'{self.inout_params.full_run_dir}/{prefix}_{stage}/combined/{self.current_epoch:04d}_{batch_idx:04d}.png'
            Path(savepath).parent.mkdir(exist_ok=True, parents=True)
            visualization_image = self.deca.visualize(visdict, savepath)
            return visdict, visualization_image[..., [2, 1, 0]]
        else:
            visualization_image = None
            return visdict, None

    @property
    def process(self):
        if not hasattr(self,"process_"):
            import psutil
            self.process_ = psutil.Process(os.getpid())
        return self.process_

import os
import cv2
import numpy as np
from pathlib import Path
from skimage.io import imsave
from .utils import torch_img_to_np, fix_image, tensor2image, upsample_mesh


def write_obj(obj_name,
              vertices,
              faces,
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    # out_normal_map = normal_map / (np.linalg.norm(
                    #     normal_map, axis=-1, keepdims=True) + 1e-9)
                    # out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(
                        normal_name,
                        # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                        normal_map
                    )
            cv2.imwrite(texture_name, texture)


def save_obj(emoca, filename, opdict, i=0):
    dense_template_path = Path(__file__).parent.parent / "assets" / "DECA" / "data" / 'texture_data_256.npy'
    dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1').item()
    vertices = opdict['verts'][i].detach().cpu().numpy()
    faces = emoca.deca.render.faces[0].detach().cpu().numpy()
    texture = tensor2image(opdict['uv_texture_gt'][i])
    uvcoords = emoca.deca.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = emoca.deca.render.uvfaces[0].detach().cpu().numpy()
    # save coarse mesh, with texture and normal map
    normal_map = tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
    write_obj(filename, vertices, faces,
                   texture=texture,
                   uvcoords=uvcoords,
                   uvfaces=uvfaces,
                   normal_map=normal_map)
    # upsample mesh, save detailed mesh
    texture = texture[:, :, [2, 1, 0]]
    normals = opdict['normals'][i].detach().cpu().numpy()
    displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
    dense_vertices, dense_colors, dense_faces = upsample_mesh(vertices, normals, faces, displacement_map, texture,
                                                                   dense_template)
    write_obj(filename.replace('.obj', '_detail.obj'),
                   dense_vertices,
                   dense_faces,
                   colors=dense_colors,
                   inverse_face_order=True)


def save_images(outfolder, name, vis_dict, i = 0, with_detection=False):
    prefix = None
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    if with_detection:
        imsave(final_out_folder / f"inputs.png",    fix_image(torch_img_to_np(vis_dict['inputs'][i])))
    imsave(final_out_folder / f"geometry_coarse.png",  fix_image(torch_img_to_np(vis_dict['geometry_coarse'][i])))
    imsave(final_out_folder / f"geometry_detail.png", fix_image(torch_img_to_np(vis_dict['geometry_detail'][i])))
    imsave(final_out_folder / f"out_im_coarse.png", fix_image(torch_img_to_np(vis_dict['output_images_coarse'][i])))
    imsave(final_out_folder / f"out_im_detail.png", fix_image(torch_img_to_np(vis_dict['output_images_detail'][i])))


def save_codes(output_folder, name, vals, i = None):
    if i is None:
        np.save(output_folder / name / f"shape.npy", vals["shapecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"pose.npy", vals["posecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"detail.npy", vals["detailcode"].detach().cpu().numpy())
    else: 
        np.save(output_folder / name / f"shape.npy", vals["shapecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"pose.npy", vals["posecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"detail.npy", vals["detailcode"][i].detach().cpu().numpy())
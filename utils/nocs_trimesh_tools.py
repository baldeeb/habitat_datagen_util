import os
import numpy as np
import shutil
import trimesh
import cv2
from quaternion import from_rotation_vector, as_rotation_matrix
import json

class MeshNocsRenderer:
    def __init__(self, resolution, hfov, glb_file):
        self.file_handle = glb_file
        self._scene = trimesh.load(glb_file)

        
        meshlist = [m for m in self._scene.geometry.values()]
        mesh_union = trimesh.util.concatenate(meshlist)
        faces = mesh_union.faces
        verts = np.array(mesh_union.vertices)
        color = self._get_nocs_colors(verts)
        verts = verts - (np.max(verts, axis=0) + np.min(verts, axis=0))/2

        # mesh = list(self._scene.geometry.values())[0]
        # color = self._get_nocs_colors(mesh.vertices)
        # verts = np.array(mesh.vertices)
        # verts = verts - (np.max(verts, axis=0) + np.min(verts, axis=0))/2
        # faces = mesh.faces
        
        # Rebuild scene with just the nocs colored object of interest.
        self._scene = trimesh.Scene(
            trimesh.Trimesh(
                vertices=verts, 
                faces=faces, 
                vertex_colors=color)
        )
        xfocal = yfocal = (resolution[1] / 2.0) / np.tan(np.deg2rad(hfov) / 2.0)
        self._scene.camera = trimesh.scene.Camera(resolution=resolution, 
                                                  focal=(xfocal, yfocal))

        # xfocal = yfocal = (resolution[0] / 2.0) / np.tan(np.deg2rad(hfov) / 2.0)
        # aspect = resolution[1]/resolution[0]
        # self._scene.camera = trimesh.scene.Camera(resolution=resolution, 
        #                                     # focal=(xfocal, yfocal),
        #                                     fov=(hfov, hfov*aspect))
        

    def _get_nocs_colors(self, vertices):
        vc = np.array(vertices) / np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
        vc -= np.min(vc, axis=0)
        vc = vc * 255
        vc = vc.astype(np.uint8)
        alpha = 255 * np.ones((len(vc), 1), dtype=np.uint8)
        return np.concatenate([vc, alpha], axis=1)

    def __call__(self, obj_pose):
        ''' returns a png image '''
        self._scene.camera
        self._scene.camera_transform = np.linalg.inv(obj_pose)
        return self._scene.save_image(visible=True)

def add_nocs_to_dataset_via_trimesh(dataloader):
    _nocs_renderers = {}
    results_dict = dataloader._data_dict
    for object_handle, file_handle in results_dict['object_handles'].items():
        _nocs_renderers[object_handle] = MeshNocsRenderer(results_dict['meta']['resolution'], 
                                                          results_dict['meta']['hfov'], 
                                                          file_handle)

    for i, eps in enumerate(results_dict['episodes']):
        # nocs_folder_name = f'{i}_nocs'
        # nocs_folder = f"{dataloader._data_dir}/{nocs_folder_name}"
        # if os.path.isdir(nocs_folder): shutil.rmtree(nocs_folder)
        # os.mkdir(nocs_folder)
        for handle, info in eps['objects'].items():
            # TODO: render every image and blend using semantic mask into one nocs image            
            if handle in _nocs_renderers:
                coord_dir = f"{dataloader._data_dir}/{i}_{handle}_coord.png"
                results_dict['episodes'][i]['objects'][handle]['coord_dir'] = coord_dir
                # results_dict['episodes'][i]['objects'][handle]['nocs_folder'] = nocs_folder
                render = _nocs_renderers[handle](info['camera_T_object'])
                buff = np.frombuffer(render, np.uint8).flatten()
                nocs = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                cv2.imwrite(coord_dir, nocs)
                # cv2.imwrite(f"{nocs_folder}/{handle}.png", nocs)
    results_dict
    f = open(f'{dataloader._data_dir}/metadata.json', 'w+')
    json.dump(results_dict, f)
    f.close()




def get_nocs_projection(mask, depth, intrinsic, camera_T_object, max_corners, 
                        min_corner, shift_by_const=True):
    '''
    mask []: used to segment depth
    depth []: 
    intrinsic []
    camera_T_object []: transform expressing the object in the camera frame.
    min/max_corner []: min and max corners
    shift_by_const (bool): instead of shifting by the min corner of the object bbox
                            shift by 0.5
    '''
    
    
    # project 2d mask to 3d
    ij = np.where(mask == 1)
    xy1 = np.vstack((ij[1], ij[0], np.ones_like(ij[0])))
    xyz = np.linalg.inv(intrinsic) @ xy1 * depth[ij]
    xyz1 = np.vstack((xyz, np.ones_like(ij[0])))
    
    # transform object to origin
    camera_T_image = np.diag([1.0,  -1.0,  -1.0, 1.0])
    object_T_image = np.linalg.inv(camera_T_object) @ camera_T_image
    centered_xyz = (object_T_image @ xyz1)[:3]

    # calculate nocs
    scale = np.linalg.norm(max_corners - min_corner)
    if shift_by_const:
        normed_dist = (centered_xyz / scale) + 0.5
    else:
        normed_dist = (centered_xyz - min_corner) / scale
    nocs_colors = (normed_dist * 255).astype(np.uint8)
    nocs = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    nocs[ij] = nocs_colors.T
    return nocs

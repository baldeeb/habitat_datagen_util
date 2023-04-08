import numpy as np


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

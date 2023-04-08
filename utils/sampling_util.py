import numpy as np

def get_patch_mask(center, image, patch_size=50):
    # calc patch position and extract the patch
    half_patch = int(patch_size / 2.0)
    low = lambda c: max(int(c - half_patch), 0)
    high = lambda c, d: min(int(c + half_patch), d)
    u_low, u_high = low(center[0]), high(center[0], image.shape[0])
    v_low, v_high = low(center[1]), high(center[1], image.shape[1])
    mask = np.zeros(image.shape[:2])
    mask[u_low:u_high, v_low:v_high] = 1
    return mask


def get_depth_mask(mean_depth, image, depth_range=20):
    m1 = np.array(image < (mean_depth+depth_range)).astype(float) 
    m2 =  np.array(image > (mean_depth-depth_range)).astype(float) 
    m3 =  np.array(image != 0).astype(float) 
    mask = m1 * m2 * m3
    return mask

    
def get_samples_from_source(rgb, depth, ij, 
                            patch_size, depth_range, 
                            max_sample_count=None):
    patch_mask = get_patch_mask(ij, rgb, patch_size)
    depth_mask = get_depth_mask(depth[ij[0], ij[1]], 
                                depth, depth_range)
    mask = depth_mask * patch_mask
    samples = np.stack(np.where(mask == 1))
    if max_sample_count:
        if samples.shape[1] != 0: 
            select = np.random.rand(max_sample_count).astype(float)
            select = (samples.shape[1] * select).astype(int)
            samples = samples[:, select]
    return samples

    
def is_in_bound(samples, shape):
    '''
    samples: [2, N]
    shape = [2]
    '''
    # print(f'samples shape: {samples}, given shape {shape}.')
    dim_checker = lambda d: [0  < s < shape[d] for s in samples[d]]
    i_list, j_list = dim_checker(0), dim_checker(1)
    return [i*j for i, j in zip(i_list, j_list)]

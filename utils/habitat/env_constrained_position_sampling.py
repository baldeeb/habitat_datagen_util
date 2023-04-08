# from habitat_tools.utils.visualizations import maps
from distancemap import distance_map_from_binary_matrix 
from quaternion import from_rotation_vector
from bresenham import bresenham
import numpy as np


# TODO: RENAME: get poses of robot facing objects
class ObjectAndRobotPoseGenerator:
    def __init__(self, sim, config):
        self.config = config
        self._m_per_px = config['_m_per_px'] 
        self._min_obj_clearence_m = config['_min_obj_clearence_m']
        self._min_rob_clearence_m = config['_min_rob_clearence_m'] 
        self._min_obj2rob_dist_m = config['_min_obj2rob_dist_m'] 
        self._max_obj2rob_dist_m = config['_max_obj2rob_dist_m'] 
        self._min_path_dist_from_obstacles = config['_min_path_dist_from_obstacles'] 
        self._sample_count = config['_sample_count'] if '_sample_count' in config else 100  # can be None
        self._height = 0 # sim.pathfinder.get_bounds()[0][1]  # get bounding box minumum
        self._height_fixed = False
        self._sim = sim


    def __call__(self):
        sim_topdown_map = self._sim.pathfinder.get_topdown_view(self._m_per_px, self._height)
        dist_map = distance_map_from_binary_matrix(np.invert(sim_topdown_map))
        min_clearence_px = np.ceil(self._min_obj_clearence_m / self._m_per_px)
        obj_px = np.stack(np.where(dist_map > min_clearence_px), axis=0)  # [2, N]
        rob_px = np.stack(np.where(dist_map > self._min_rob_clearence_m), axis=0)  # [2, N]

        if obj_px.shape[1] == 0 :
            raise RuntimeError('Cound not find valid positions to place object...')
        if rob_px.shape[1] == 0:
            raise RuntimeError('Cound not find valid positions to place robot...')

        obj_px, rob_px = self._sample_pairs(obj_px, rob_px, count=self._sample_count)
        dist_map_m = dist_map * self._m_per_px
        obj_px, rob_px = self._filter_sampled_rob2obj_points(obj_px, rob_px, dist_map_m)
        
        obj_xyz = self._ij_to_xyz(obj_px)  # [3, M]
        rob_xyz = self._ij_to_xyz(rob_px)  # [3, M]
        rob_thetas = np.array( [ self._get_theta_a2b(a, b) \
                for a, b in zip(rob_xyz.T, obj_xyz.T)])  #[M,]
        rob_rot = [from_rotation_vector([0.0, t, 0.0]) 
                   for t in rob_thetas] # to quaternions
        return rob_xyz, rob_rot, obj_xyz

    def _sample_pairs(self, obj_pxs, rob_pxs, count=None):
        # Sample from each set of candidates
        if count is None:
            rob_count, obj_count = rob_pxs.shape[1], obj_pxs.shape[1]
        else:
            count = min([rob_pxs.shape[1], obj_pxs.shape[1], count])
            rob_count = obj_count = count
            def choose(candidates, count):
                idx_range = np.arange(candidates.shape[1])
                idxs = np.random.choice(idx_range, size=count, replace=False)
                return candidates[:, idxs]
            rob_pxs, obj_pxs = choose(rob_pxs, count), choose(obj_pxs, count)
        # Sample pairs
        pair_idxs = np.mgrid[:rob_count, :obj_count].reshape(2, -1)
        paired_rob_pxs = rob_pxs[:, pair_idxs[0, :]]
        paired_obj_pxs = obj_pxs[:, pair_idxs[1, :]]
        return paired_obj_pxs, paired_rob_pxs


    # Filter unwanted pairs
    def _filter_sampled_rob2obj_points(self, obj_pxs, rob_pxs, dist_map_m):
        ''' 
        Given associated robot and object pixel locations,
        this code returns the pixels that are far apart yet
        have a clear line of sight between them
        '''
        # Remove pairs that are too close
        min_rob2obj_dist_px = self._min_obj2rob_dist_m / self._m_per_px
        max_rob2obj_dist_px = self._max_obj2rob_dist_m / self._m_per_px
        px_dist = np.linalg.norm(obj_pxs - rob_pxs, axis=0)
        sparse_idxs = np.stack(np.where((min_rob2obj_dist_px <= px_dist) & (px_dist <= max_rob2obj_dist_px)), axis=0)  # 2 x N  
        # Remove pairs with obstructed view
        unobstructed_idxs = []
        for i in sparse_idxs.squeeze():
            line = np.array(list(bresenham(
                    int(obj_pxs[0, i]), int(obj_pxs[1, i]), 
                    int(rob_pxs[0, i]), int(rob_pxs[1, i]))))
            if all(dist_map_m[line[:, 0], line[:, 1]] >= self._min_path_dist_from_obstacles):
                unobstructed_idxs.append(i)
        return obj_pxs[:, unobstructed_idxs], rob_pxs[:, unobstructed_idxs]


    def _ij_to_xyz(self, ij):
        bounds = np.array(self._sim.pathfinder.get_bounds())
        zx = ij * self._m_per_px + bounds[0, [2, 0], np.newaxis] # [2, N]
        return np.stack([zx[1], self._height * np.ones(zx.shape[1]), zx[0]], axis=0) # [3, N]
        

    def _get_theta_a2b(self, a, b):
        ''' Given 2d points find theta that directs a towards b'''
        a, b  = np.array(a), np.array(b)
        c  = a - b
        theta = np.arctan2(c[0], c[2])
        return theta
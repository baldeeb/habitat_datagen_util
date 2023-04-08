import os
import cv2
import json
import shutil
import quaternion
import numpy as np
import magnum as mn
from tqdm import tqdm
from utils.habitat.dataset import HabitatDataloader
from utils.habitat.nocs_tools import get_nocs_projection
from utils.habitat.env_constrained_position_sampling  import ObjectAndRobotPoseGenerator


def get_intrinsic_from_habitat_sensor_spec(spec):
    Ho2, Wo2 = (spec.resolution[i]/2.0 for i in range(2))
    hfov = float(mn.Rad(spec.hfov))
    fy_inv = fx_inv = Wo2 / np.tan(hfov / 2.)
    return np.array([[fx_inv,    0.,        Wo2],
                    [0.,         fy_inv,    Ho2],
                    [0.,         0.,        1.0]])


class SimulatorPhysicsStepFunctor:
    def __init__(self, sim, config):
        self._sim, self._config = sim, config
        self._t_end = self._sim.get_world_time() + self._config['duration']
        self._dt = 1.0 / self._config['frequency']


    def __len__(self):
        # TODO: FIX THIS. always constant
        t_remaining = self._t_end - self._sim.get_world_time()
        return int( t_remaining * self._config['frequency'])
    
    def __call__(self):
        if 'action' in self._config:
            raise RuntimeError("not yet implemented.")
            self._sim.step(self._config['action'], dt=self._dt)
        else:
            self._sim.step_physics(self._dt)

    def __iter__(self): 
        return self
    def __next__(self):
        if len(self) == 0: raise StopIteration
        return self()


class SimulatorPoseTeleportationFunction:
    def __init__(self, sim, config):
        self._sim, self._config = sim, config

        self.rob_xyz, self.rob_t, self.obj_xyz = \
            ObjectAndRobotPoseGenerator(sim, config)()
        assert(self.rob_xyz.shape[1] == len(self.rob_t) == self.obj_xyz.shape[1])
        if 'sample_count' in config:
            top_idx = min(config['sample_count'], self.rob_xyz.shape[1])
            self.rob_xyz = self.rob_xyz[:, :top_idx]
            self.rob_t = self.rob_t[:top_idx]
            self.obj_xyz = self.obj_xyz[:, :top_idx]

        rom = sim.get_rigid_object_manager()
        handles = config['object_file_handles']
        self._objects = [rom.get_object_by_handle(h) for h in handles]
        self._idx = 0

    def __len__(self):
        return self.rob_xyz.shape[1]
    
    def __call__(self):
        if self._idx == (len(self)): raise StopIteration
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        state.position = self.rob_xyz[: , self._idx]
        state.rotation = self.rob_t[self._idx]
        agent.set_state(state)
        
        for obj in self._objects:
            # TODO: have option to drop from height
            # TODO: Make it capable of handling multiple objects
            # TODO: add noise to pose
            obj_position = self.obj_xyz[:, self._idx].copy()
            obj_position[1] -= obj.collision_shape_aabb.min[1]

            # obj_position[1] += 0.75 # TODO: remove. offsets object to stay on ground
            
            # Adds gaussian noise to the object position
            obj_pose_noise = np.random.normal(
                np.array([0.0, 0.0, 0.0]), 
                np.array([0.25 * self._config['_min_obj_clearence_m'],
                    0.0, 0.25 * self._config['_min_obj_clearence_m']]))
            obj.translation = obj_position + obj_pose_noise

            # Set rotation
            # TODO: make this configurable
            rand_theta = float(2*np.pi*np.random.rand())
            obj.rotation = mn.Quaternion.rotation(mn.Rad(rand_theta), mn.Vector3([0, 1, 0]))

            if 'simulate_gravity_for_s' in  self._config:
                self._sim.step_physics(self._config['simulate_gravity_for_s'])
        self._idx += 1

    def __iter__(self): 
        return self
    def __next__(self):
        if len(self) == 0: raise StopIteration
        return self()


class SimulationDataStreamer:
    '''
    Simulates and returns one view at a time
    for now this class expects the user to initialize and set motion.
    '''
    def __init__(self, simulator, config):
        self._config = config
        self._sim = simulator
        if config['step_type'] == 'physics':
            self._step = SimulatorPhysicsStepFunctor(self._sim, config)
        elif config['step_type'] == 'teleport':
            self._step = SimulatorPoseTeleportationFunction(simulator, config)    
            self._step()

        self._obj_handles_dict = {} 
        if 'object_file_handles' in config:
            obj_mngr = self._sim.get_rigid_object_manager()
            for o in config['object_file_handles']:
                obj = obj_mngr.get_object_by_handle(o)
                file_handle = obj.creation_attributes.collision_asset_handle
                self._obj_handles_dict[obj.handle] = file_handle
        self._K = self.intrinsic()


    def intrinsic(self):
        sensor_name = self._config['rgb_sensor']
        spec = self._sim._sensors[sensor_name]._spec
        return get_intrinsic_from_habitat_sensor_spec(spec)

    def __len__(self):
        return len(self._step)

    def _to_matrix(self, q, t):
        '''q: quaternion
            t: translation'''
        m = np.eye(4)
        m[0:3, 0:3] = quaternion.as_rotation_matrix(q)
        m[0:3, 3] = t
        return m

    def __getitem__(self, _):
        if len(self) <= 0: return None
        
        results = {}

        obs = self._sim.get_sensor_observations()
        agent_state = self._sim.agents[0].get_state()
        
        results['rgb'] = obs[self._config['rgb_sensor']][:, :, 0:3]
        results['depth'] = obs[self._config['depth_sensor']]
        if 'semantic_sensor' in self._config:
            results['semantics'] = obs[self._config['semantic_sensor']]
        
        # Save agent info
        wTa = results['world_T_agent'] = self._to_matrix(agent_state.rotation, agent_state.position)
        sensor_state = agent_state.sensor_states['color_sensor_1st_person']
        wTs = results['world_T_rgb'] = self._to_matrix(sensor_state.rotation, sensor_state.position)

        # Save static objects' info
        obj_info = {}
        manager = self._sim.get_rigid_object_manager()
        for handle in manager.get_object_handles():
            obj = manager.get_object_by_handle(handle)
            wTo = obj.transformation
            sTo = np.linalg.inv(wTs) @ wTo
            # aTo = np.linalg.inv(wTa) @ wTo

            obj_info[handle] = {
                # 'world_T_object': wTo, 'agent_T_object': aTo,
                'camera_T_object': sTo, 'world_T_rgb': wTs,
                'camera_T_image': np.diag([1.0,  -1.0,  -1.0, 1.0]),
                'semantic_id': obj.semantic_id,
                'class': obj.user_attributes.get('class'),
                'bbox':{'min': obj.collision_shape_aabb.min,
                        'max': obj.collision_shape_aabb.max}
                }
                
            aabb_min, aabb_max = obj.collision_shape_aabb.min, obj.collision_shape_aabb.max
            mask = np.zeros_like(results['semantics'])
            mask[results['semantics'] == obj.semantic_id] = 1
            reshape_corner = lambda a :np.array(a).reshape((3, 1))
            coord = get_nocs_projection(mask, results['depth'], self._K, sTo, 
                                        reshape_corner(aabb_max), reshape_corner(aabb_min))
            obj_info[handle]['coord'] = coord

        results['objects'] = obj_info
        self._step()
        return results    

    def __iter__(self): 
        return self

    def __next__(self):
        if len(self) == 0: raise StopIteration
        return self[None]

    def save(self, folder, override=False):

        print("Saving data to {}".format(folder))
        
        # Check folder
        if os.path.exists(folder): 
            if override: shutil.rmtree(folder)
            else: raise RuntimeError("Can't override existing path.")
        os.makedirs(folder)

        # Helper functions
        nparray2list = lambda arr: [float(a) for a in arr]
        matrix2d2list = lambda matt: [nparray2list(r) for r in matt]

        # Set up data dictionary
        data_dict = {'episodes':[]}
        data_dict[ "source"] = "habitat"
        data_dict['meta'] = {
            'camera_intrinsic' : matrix2d2list(self.intrinsic()),
            'img_format' : 'png', #'npy',
        }

        # Record sensor specs
        sensor_spec = self._sim._sensors['color_sensor_1st_person']._spec
        data_dict['meta']['hfov'] = float(sensor_spec.hfov)
        data_dict['meta']['resolution'] = nparray2list(sensor_spec.resolution[-1::-1])

        # Step and record data
        for i, results in tqdm(enumerate(self), total=len(self)):

            obj_info = {}
            coord = np.zeros_like(results['rgb'])
            for k, v in results['objects'].items():
                obj_info[k] = {
                    # 'world_T_object':matrix2d2list(v['world_T_object']),
                    # 'agent_T_object':matrix2d2list(v['agent_T_object']),
                    'camera_T_image':matrix2d2list(v['camera_T_image']),
                    'camera_T_object':matrix2d2list(v['camera_T_object']),
                    'world_T_rgb': matrix2d2list(v['world_T_rgb']),
                    'semantic_id': v['semantic_id'],
                    'bbox':{'min': nparray2list(v['bbox']['min']), 
                            'max': nparray2list(v['bbox']['max'])}
                }
                if 'coord' in v:
                    # coord = cv2.cvtColor(v['coord'], cv2.COLOR_RGB2BGR)
                    # coord += cv2.cvtColor(v['coord'], cv2.COLOR_RGB2BGR)
                    coord += v['coord']


            # Record episode
            episode = {
                'color':f'{i}_color',
                'depth':f'{i}_depth',
                'mask':f'{i}_mask',
                'coord':f'{i}_coord',
                'objects': obj_info,
                'word_T_agent': matrix2d2list(results['world_T_agent'])
            }
            
            rgb = cv2.cvtColor(results['rgb'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{folder}/{episode['color']}.png", 
                        rgb.astype(np.uint8))
            depth = results['depth'] * 10**3
            cv2.imwrite(f"{folder}/{episode['depth']}.png", 
                        depth.astype(np.uint16))
            if 'semantics' in results:
                cv2.imwrite(f"{folder}/{episode['mask']}.png", 
                            results['semantics'].astype(np.uint8))
            cv2.imwrite(f"{folder}/{episode['coord']}.png",
                        coord.astype(np.uint8))

            with open(f"{folder}/{i}_meta.txt", 'w+') as f:
                for io, (k, v) in enumerate(results['objects'].items()):
                    if 'object_models_path' in self._config:
                        obj_path = os.path.relpath(self._obj_handles_dict[k],
                                                   self._config['object_models_path'])
                        f.write(f"{v['semantic_id']} {v['class']} {obj_path}\n")
                    elif 'object_file_handles' in self._config:
                        obj_path = self._obj_handles_dict[k]
                        f.write(f"{v['semantic_id']} {v['class']} {obj_path}\n")

            data_dict['episodes'].append(episode)
        data_dict['object_handles'] = self._obj_handles_dict

        # Save data in file
        f = open(f'{folder}/metadata.json', 'w+')
        json.dump(data_dict, f)
        f.close()
        return HabitatDataloader(f'{folder}/metadata.json') 




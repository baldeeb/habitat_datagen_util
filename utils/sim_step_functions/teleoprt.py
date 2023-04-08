import numpy as np
import magnum as mn
from utils.viewpoint_generator  import ObjectAndRobotPoseGenerator


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
            obj.translation = obj_position #+ obj_pose_noise

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


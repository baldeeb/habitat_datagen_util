
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
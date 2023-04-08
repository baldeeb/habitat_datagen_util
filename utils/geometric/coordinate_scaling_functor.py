import numpy as np

class CoordScalingFunctor:
    def __init__(self, source_shape, target_shape):
        self._src, self._tgt = source_shape, target_shape
        self.scale = [(t/s)for s,t  in zip(source_shape, target_shape)]

    def __call__(self, ij):
        ''':param ij: shape [2, N]''' 
        return np.array([(i*s).astype(int) for i, s in zip(ij, self.scale)])
        
    def __str__(self):
        return f'source {self._src},  target {self._tgt},  scale {self.scale}'

    def inverse(self): 
        '''returns a copy of class with inverse function'''
        return CoordScalingFunctor(self._tgt, self._src)
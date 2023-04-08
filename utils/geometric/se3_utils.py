import torch 


def homogenize(s): 
    return torch.cat([s, torch.ones((1, s.shape[1]))], dim=0)    

class Se3PixelHandler:
    """Used to return pixels after they are transformed in SE3"""
    def __init__(self, n_ij_samples, sample_depths, intrinsic):
        """
        n_ij_samples [2, N]
        sample depths [N]
        intrinsic [3, 3]
        """
        self.K = torch.tensor(intrinsic).float()
        self.K_inv = self.K.inverse().float()
        self.n = len(sample_depths)
        n_ij = torch.tensor(n_ij_samples).float()
        self.n_uv = n_ij.index_select(0, torch.LongTensor([1,0]))
        self.n_d = torch.tensor(sample_depths).float()

        self.xyz = self.K_inv @ homogenize(self.n_uv) * self.n_d
        self.xyz1 = homogenize(self.xyz)
    
    
    def get_3d(self, se3:torch.tensor=None):
        if se3 is not None:
            transformed = se3 @ self.xyz1.clone()
            return transformed[0:3, :]
        else:
            return self.xyz.clone()

    def __len__(self): return self.n

    def __call__(self, se3:torch.tensor=torch.eye(4)):
        ''' returns ij indices of projected pixels after transform is applied'''
        transformed = se3 @ self.xyz1.clone()
        projected = self.K @ (transformed[0:3, :] / transformed[2, :])
        return projected.index_select(0, torch.LongTensor([1,0]))  


class DifferentiableSE3transform(torch.nn.Module):
    def __init__(self, rx=0., ry=0., rz=0., tx=0., ty=0., tz=0.):
        super(DifferentiableSE3transform, self).__init__()
        Parameters = torch.nn.parameter.Parameter
        self.rots = Parameters(torch.FloatTensor([[rx], [ry], [rz]]))
        self.trans = Parameters(torch.FloatTensor([tx, ty, tz]))
        
    def _rx(self):
        t = torch.FloatTensor
        c, s = torch.cos(self.rots[0]), torch.sin(self.rots[0])
        return  torch.cat([
                    torch.cat([t([1.]),  t([0.]), t([0.]) ]).unsqueeze(0),
                    torch.cat([t([0.]),  c,       s,      ]).unsqueeze(0),
                    torch.cat([t([0.]),  -s,      c,      ]).unsqueeze(0)])
    def _ry(self):
        t = torch.FloatTensor
        c, s = torch.cos(self.rots[1]), torch.sin(self.rots[1])
        return torch.cat([
                    torch.cat([c,       t([0.]), s        ]).unsqueeze(0),
                    torch.cat([t([0.]), t([1.]), t([0.])  ]).unsqueeze(0),
                    torch.cat([-s,      t([0.]), c        ]).unsqueeze(0)])
    def _rz(self):
        t = torch.FloatTensor
        c, s = torch.cos(self.rots[2]), torch.sin(self.rots[2])
        return  torch.cat([
                    torch.cat([c,       s,       t([0.]) ]).unsqueeze(0),
                    torch.cat([-s,      c,       t([0.]) ]).unsqueeze(0), 
                    torch.cat([t([0.]), t([0.]), t([1.]) ]).unsqueeze(0)])

    def __call__(self):
        bottom = torch.tensor([[0., 0., 0., 1.]])
        right = self.trans.unsqueeze(1)
        rotation = self._rx() @ self._ry() @ self._rz()
        top = torch.cat([rotation, right], axis=1)
        se3 = torch.cat([top, bottom], axis=0)
        return se3

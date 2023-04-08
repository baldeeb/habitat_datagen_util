import os
import json
import numpy as np
import cv2


# import pathlib
class HabitatDataloader:
    def __init__(self, meta_file):
        
        # TODO: allow this to take in folder with data
        #   - will need to adjust the length to account for all subfolders
        #   - and need to select the subfolder based on i given to getitem
        # p = pathlib.Path(directory)
        # metadata_files = p.rglob('*.json')

        self._data_dir = os.path.dirname(meta_file)
        f = open(meta_file, 'r')
        self._data_dict = json.load(f)

    def intrinsic(self):
        return np.array(self._data_dict['meta']['camera_intrinsic'])

    def __len__(self):
        return len(self._data_dict['episodes'])

    def __getitem__(self, i):
        e = self._data_dict['episodes'][i]
        format = self._data_dict['meta']['img_format']
        rgb_f = f"{self._data_dir}/{e['color']}.{format}"
        d_f = f"{self._data_dir}/{e['depth']}.{format}"
        s_f = f"{self._data_dir}/{e['mask']}.{format}" if 'mask' in e else None
        c_f = f"{self._data_dir}/{e['coord']}.{format}" if 'mask' in e else None

        if format == 'npy':
            rgb = np.load(rgb_f)
            d = np.load(d_f)
            s = np.load(s_f) if s_f else None
        else:
            rgb = cv2.imread(rgb_f, cv2.IMREAD_ANYCOLOR) 
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            d = cv2.imread(d_f, cv2.IMREAD_ANYDEPTH).astype(np.float32) * 10**(-3)
            s = cv2.imread(s_f, cv2.IMREAD_ANYDEPTH).astype(np.uint8)
            coord = cv2.imread(c_f, cv2.COLOR_BGR2RGB)
            # coord = cv2.cvtColor(coord, cv2.COLOR_BGR2RGB)

        return rgb, d, s, coord, e

    def get_transform(self, i, obj_handle): 
        e = self._data_dict['episodes'][i]
        obj = e['objects'][obj_handle]
        return np.array(obj['transformation'])

    def get_relative_transform(self, source, target, obj_handle=None):
        if obj_handle is None:
            obj_handle = list(self._data_dict['episodes'][source]['objects'].keys())[0]
        T_s = self.get_transform(source, obj_handle)
        T_t = self.get_transform(target, obj_handle)
        return T_t@np.linalg.inv(T_s)

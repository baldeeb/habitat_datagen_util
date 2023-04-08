
import os
import csv
import shutil
import numpy as np
import pathlib as pl

from utils.datagen_utils import (add_metadata_to_object_dir,
                                 files_generator)


def get_target_folders_in_path(path, targets):
    '''Generator that yields the full path of all target folders in path.'''
    seen = {}
    for dir, folders, _ in os.walk(path):
        dir = os.path.abspath(dir)
        for f in folders:
            if f in targets and f not in seen:
                seen[f] = True
                yield os.path.join(dir, f)

def read_shapenet_meta_file(file_path):
    '''Returns file as dict with rows as lists.'''
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = {r:list() for r in next(reader)}
        for row in reader:
            for k, v in zip(data.keys(), row):
                data[k].append(v)
    return data

def get_shapenet_obj_ids_from_meta_file(file_path):
    '''Returns only the fullIds of the objects in the meta file.'''
    data = read_shapenet_meta_file(file_path)
    return [d.split('.')[-1] for d in data['fullId']]


def create_symlinks(output_folder, path_list):
    '''Create folder with links to provided paths.'''
    if os.path.isdir(output_folder): 
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    for f in path_list:
        link_name = output_folder / pl.Path(f).name
        os.symlink(f, link_name)


class ShapenetObjectHandler(object):
    '''Designed to dispernse specified object ids until the provided list is exhausted.
    Afterwards, random objects are dispensed.'''
    def __init__(self, root, obj_ids, class_ids,
                 format='glb', get_cfg=True,
                 shuffle=False, seed=None,
                 default_obj_metadata={
                     "mass": 0.95, 
                     "render_asset": 'model_normalized.glb',}
                 ):
        self.root = pl.Path(root)
        assert(self.root.is_dir())

        self._obj_ids = obj_ids
        self._default_meta = default_obj_metadata

        if isinstance(class_ids, list):
            self._classes = class_ids  
        else: 
            self._classes = np.ones_like(obj_ids, dtype=int) * class_ids
        
        self._get_cfg = get_cfg
        self._format = format

        self._idxs = np.arange(len(obj_ids))
        folders = get_target_folders_in_path(self.root, obj_ids)
        self._files = [next(files_generator(f, format)) for f in folders]
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self._idxs)
        self._counter = 0

    def _config_of(self, obj_file): 
        cfg = obj_file[:-3] + 'object_config.json'
        if not pl.Path(cfg).is_file(): 
            add_metadata_to_object_dir(obj_file, 
                                       self._format,
                                       self._default_meta) 
        return cfg

    def _process_idx(self, idx):
        if idx is None:
            idx = self._counter
            self._counter += 1
            # idx = np.random.choice(len(self._objs))
        if idx >= len(self._idxs):
            idx = np.random.choice(len(self._idxs))
        return self._idxs[idx]

    def __getitem__(self, idx=None):
        i = self._process_idx(idx)
        pth = self._files[i]
        if self._get_cfg: pth = self._config_of(pth)
        return pth, self._classes[i]

    def __len__(self):
        return 1
        # return len(self._objs)
        
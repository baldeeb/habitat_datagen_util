
import os
import csv
import shutil
import pathlib

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
        link_name = output_folder / pathlib.Path(f).name
        os.symlink(f, link_name)

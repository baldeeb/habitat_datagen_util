import os
import json
from glob import glob
from pathlib import Path
from os.path import relpath

def files_generator(directory, extension='obj'):
    for obj_f in glob(f'{directory}/**/**.{extension}', recursive=True):
        yield obj_f

def folders_generator(directory):
    for obj_f in files_generator(directory):
        yield os.path.dirname(obj_f)

def generate_object_directory_metadata_file(file_paths):
    directory = Path(file_paths[0]).parent
    with open(directory / 'metadata.txt', 'w') as f:
        for p in file_paths:
            f.write(f'{relpath(p, directory)}\n')

DEFAULT_HABITAT_OBJ_CONFIG={
    "mass": 0.95,
    "render_asset": 'model_normalized.obj',
}

def create_habitat_object_config_file(file_path, config_dict=DEFAULT_HABITAT_OBJ_CONFIG):
    file_path = Path(file_path)
    config_dict['render_asset'] = file_path.name
    with open( f'{file_path.parent / file_path.stem}.object_config.json', 'w') as f: 
        json.dump(config_dict, f, indent=4)

def add_metadata_to_object_dir(directory, extension='obj', config_dict=DEFAULT_HABITAT_OBJ_CONFIG):
    file_paths =  list(files_generator(directory, extension))
    generate_object_directory_metadata_file(file_paths)
    for p in file_paths: create_habitat_object_config_file(p, config_dict)
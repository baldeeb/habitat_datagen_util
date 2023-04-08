import os
import json
import trimesh
import argparse
import numpy as np

class ObjToGlb:
    def __init__(self, obj_path, glb_path, config_dict=None, denormalize=False, override=False):
        self._config = config_dict
        self._denorm = denormalize
        self._override = override
        self.glb_files_added = []
        self._run(obj_path, glb_path)
        if os.path.isdir(obj_path): obj_path_folder = obj_path
        else: obj_path_folder = os.path.dirname(obj_path)
        with open(obj_path_folder+'/glb_file_list.txt', 'w+') as f: 
            pl = len(obj_path_folder)
            str_list = ''.join([f'{x[pl:]}\n' for x in self.glb_files_added])
            f.write(str_list)

    def _run(self, obj_path, glb_path):
        if os.path.isdir(obj_path):
            assert(os.path.isdir(glb_path))
            for subpath in os.listdir(obj_path):
                full_subpath = os.path.join(obj_path, subpath)
                if os.path.isdir(full_subpath): 
                    self._run(full_subpath, os.path.join(glb_path, subpath))
                elif subpath[-4:] == '.obj': 
                    self._single_obj(full_subpath, glb_path+'/'+subpath[0:-3]+'glb')
        else: 
            assert(obj_path[-4:] == '.obj' and glb_path[-4:] == '.glb')
            self._single_obj(obj_path, glb_path)

    def _single_obj(self, obj, glb):
        print(f'converting:\n\t<---{obj}\n\t-->{glb}')
        self.glb_files_added.append(glb)
        mesh = trimesh.load(obj)
        if self._denorm:
            with open(obj[:-4]+'.json', 'r') as f: 
                obj_config = json.load(f)
                min, max, center = (np.array(obj_config[k]) 
                                    for k in ['min', 'max', 'centroid'])
                mesh.apply_scale(np.linalg.norm(max - min))
                mesh.apply_translation(center)
        if self._override and os.path.isfile(glb): os.remove(glb)
        mesh.export(glb, file_type='glb')
        # data = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
        # with open(glb, 'wb') as f: 
        #     f.write(data)
        if self._config:
            config_dict = self._config.copy()
            config_dict['render_asset'] = glb.split('/')[-1]
            with open(glb[:-4] + '.object_config.json', 'w') as f: 
                json.dump(config_dict, f, indent=4)
            print(f'\n\tadded config file: {config_dict["render_asset"]}')


def obj_to_glb(args):
    if args.target is None: args.target = args.source
    config_dict = {'mass':0.95} if args.add_config else None
    ObjToGlb(args.source, args.target, config_dict, args.denormalize, args.override)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--source', type=str, default='', 
                        help='source path', 
                        required=True)
    parser.add_argument('-t', '--target', type=str, default=None, 
                        help='target path. If not set, the source folder will be used as target.', 
                        required=False)
    parser.add_argument('--add-config', action='store_true',
                        help='creates default configuration json files used by the Habitat simulator.')
    parser.add_argument('--denormalize', action='store_true',
                        help='creates default configuration json files used by the Habitat simulator.')
    parser.add_argument('--override', action='store_true',
                        help='remove glb versions of the obj model if they exist.')
    args = parser.parse_args()
    obj_to_glb(args)

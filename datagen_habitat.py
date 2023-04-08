import os
from os.path import join as joinpath
import pathlib as pl

import magnum as mn
import numpy as np
import habitat_sim
from utils.cfg_helpers import make_cfg
from utils.sim_streamer import SimulationDataStreamer
from utils.viewpoint_generator import ObjectAndRobotPoseGenerator
import yaml
from utils.shapenet_tools import ShapenetObjectHandler
import time 

# Get current file path
current_file_path = pl.Path(__file__).parent.absolute()

# Config
yaml_path = current_file_path/'config/habitat_datagen.yaml'
with open(yaml_path) as f: args = yaml.load(f, Loader=yaml.CLoader)

# Set data paths
DOWNLOADED_DATA_DIR = current_file_path / 'data/downloaded'
SCENE_DIR = joinpath(DOWNLOADED_DATA_DIR, 'scene_datasets/hm3d/val')
SAVE_TO_FOLDER = current_file_path / pl.Path(args['output']['relative_directory'])

# Set scenes to use
scene_config = f'{SCENE_DIR}/hm3d_val_basis.scene_dataset_config.json'  # We needed a non-anotated scene 
                                                                        # to control semantic annotations
scene_glb_list = [os.path.join(SCENE_DIR, scene, f'{scene.split("-")[-1]}.{args["objects"]["format"]}') 
                  for scene  in os.listdir(SCENE_DIR)
                  if os.path.isdir(os.path.join(SCENE_DIR, scene))]
scene_glb_list = np.random.choice(scene_glb_list, 
                                  size=args['simulation']['num_scenes_to_use'], 
                                  replace=args['simulation']['num_scenes_to_use']>len(scene_glb_list))

object_handler = ShapenetObjectHandler(root=args['objects']['direcotry'],
                                       obj_ids=args['objects']['chairs']['test'],
                                       class_ids=7, shuffle=True,)

def random_camera_height_and_pitch():
    sensor_args = args['simulation']['sensor']
    h = np.random.uniform(sensor_args['height_range'][0], 
                          sensor_args['height_range'][1])
    p = -np.arctan2(h, sensor_args['distance_to_fixate_on'])
    p += np.random.uniform(-sensor_args['pitch_noise'], 
                           sensor_args['pitch_noise'])
    return h, p


for scene_idx in range(len(scene_glb_list)):
    print(f'Generating data for scene {scene_idx+1} out of {len(scene_glb_list)}')
    scene_glb = scene_glb_list[scene_idx]
    # object_folders = scene_obj_folders[scene_idx]
    object_folders, object_class = object_handler[scene_idx]
    camera_height, camera_pitch = random_camera_height_and_pitch()

    # Generate config
    args['simulation']['simulator'].update({
        "scene":  scene_glb,
        "scene_dataset_config": scene_config,
        "sensor_height": camera_height,
        "sensor_pitch": camera_pitch,
    })
    cfg = make_cfg(args['simulation']['simulator'])


    # Create simulator
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print('#'*50, e, '#'*50); continue

    # Get object managers
    obj_attr_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()

    if len(object_folders) > 1:
        print(f'NOTICE: more than one object folder was provided, only the first will be used')

    obj_idx, obj_f = 0, object_folders  # TODO: make a for loop to handle many objects 
    obj_attr_mgr.load_configs(obj_f)
    f_handle = obj_attr_mgr.get_file_template_handles(obj_f)[0]
    obj = rigid_obj_mgr.add_object_by_template_handle(f_handle)
    obj.semantic_id = obj_idx + 1
    obj.user_attributes.set('class', object_class)

    common_dataloader_configs = {
        'object_file_handles': rigid_obj_mgr.get_object_handles(),

        # When set, the streaming class saves relative object locations 
        #   in a meta file instead of the complete location
        'object_models_path': os.path.join(DOWNLOADED_DATA_DIR,'objects'),  

        'rgb_sensor':'color_sensor_1st_person',
        'depth_sensor':'depth_sensor_1st_person',
        'semantic_sensor':'semantic_sensor_1st_person'
    }


    # Setup to generate data by teleporting the agent
    if args['simulation']['mode'] == 'Teleport':

        # Instantiate dataloader
        sim_dataloader_config = {
            # step config
            'step_type': 'teleport',
            '_m_per_px': 0.25, 

            # distance of object form obstacles
            '_min_obj_clearence_m': 0.7,    
            
            # distance of robot form obstacles
            '_min_rob_clearence_m': 0.2,   # NOTE: Only first floor for now
            '_min_obj2rob_dist_m': 1.5  ,
            '_max_obj2rob_dist_m': 4,
            '_min_path_dist_from_obstacles': 0.2,

            '_sample_count': args['simulation']['image_count'],              
        }
        sim_dataloader_config.update(common_dataloader_configs)


    # Setup to generate data by moving the objects
    elif args['simulation']['mode'] == 'Kinematic':
        
        print('#'*50, "\nKinematic data generation is not fully setup yet in this script.\n",'#'*50)
        pose_gen_config = {
            '_m_per_px': 0.1, 
            '_min_obj_clearence_m': 0.5,   # distance of object form obstacles 
            '_min_rob_clearence_m': 0.2,   # distance of robot form obstacles   
                                           # NOTE: Only first floor for now
            '_min_obj2rob_dist_m': 1.5  ,
            '_max_obj2rob_dist_m': 4  ,
            '_min_path_dist_from_obstacles': 0.2,
            '_sample_count': args['simulation']['image_count'],
        }
        rob_xyz, rob_t, obj_xyz = ObjectAndRobotPoseGenerator(sim, pose_gen_config)()

        # Set robot state
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = rob_xyz[:, 0]
        state.rotation = rob_t[0]
        agent.set_state(state)

        # Set object state
        obj_handle = rigid_obj_mgr.get_object_handles()[0]
        obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = obj_xyz[:, 0] - [0, obj.collision_shape_aabb.min[1], 0]
        obj.rotation = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3([1,0,0]))
        obj.velocity_control.controlling_ang_vel = True
        obj.velocity_control.linear_velocity = mn.Vector3((0.0, 0.0, 0.0))
        obj.velocity_control.angular_velocity = mn.Vector3((0.0, 0.2, 0.0))

        # Instantiate dataloader
        sim_dataloader_config = {
            'step_type': 'physics',
            'duration':20.0, 
            'frequency':10.0,
        }
        sim_dataloader_config.update(common_dataloader_configs)

    # Run the data generator
    try:
        dataloader = SimulationDataStreamer(sim, sim_dataloader_config)
        scene_name = scene_glb.split('/')[-2] + '_' + time.time().__str__()
        save_to_sub_folder = os.path.join(SAVE_TO_FOLDER, scene_name)
        dataloader.save(save_to_sub_folder, override=args['output']['overwrite'])
        print("\nData Generated Successfully!")
    except StopIteration as e:  # TODO: Make custom exception
        print("Failed to generate and save data...")
        print(e)
    except RuntimeError as e:  # TODO: Make custom exception
        print("Failed to generate and save data...")
        print(e)
    sim.close()
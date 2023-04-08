# %% TODO list
# - [x] Add the ability to generate a dataset per scene
# - [x] Support more than one and different combinations of objects
# - [x] Allow the teleport-datagen algo to use multiple floors

# %% Import all that is needed
import os
from os.path import join as joinpath
import pathlib as pl

import magnum as mn
import numpy as np
import habitat_sim
from utils.habitat.default_helper_functions import make_cfg
from utils.habitat.datatools import SimulationDataStreamer
from utils.habitat.env_constrained_position_sampling import ObjectAndRobotPoseGenerator
from utils.habitat.datagen_utils import (obj_folders_generator, 
                                               add_metadata_to_object_dir,
                                               obj_files_generator)
import yaml
from utils.shapenet_tools import get_target_folders_in_path

# Get current file path
current_file_path = pl.Path(__file__).parent.absolute()

# Config
yaml_path = current_file_path/'config/habitat_datagen.yaml'
with open(yaml_path) as f: args = yaml.load(f, Loader=yaml.CLoader)
OBJECTS_DIR = args['objects']['direcotry']


# Set data paths
OVERWRITE_EXISTING_DATA = False
OUTPUT_FOLDER_NAME = '200of100scenes_26selectChairs/val'
GEN_METHOD = 'Teleport' #'Teleport', 'Kinematic'
NUM_SCENES_TO_USE = 5

OUTPUT_DIR = current_file_path / 'data/generated'
DOWNLOADED_DATA_DIR = current_file_path / 'data/downloaded'
SCENE_DIR = joinpath(DOWNLOADED_DATA_DIR, 'scene_datasets/hm3d/val')
# OBJECTS_DIR = joinpath(DOWNLOADED_DATA_DIR, 'objects/swivel_chair_models')
SAVE_TO_FOLDER = joinpath(OUTPUT_DIR, OUTPUT_FOLDER_NAME)


# Set scenes to use
scene_config = f'{SCENE_DIR}/hm3d_val_basis.scene_dataset_config.json'  # We needed a non-anotated scene 
                                                                        # to control semantic annotations
scene_glb_list = [os.path.join(SCENE_DIR, scene, f'{scene.split("-")[-1]}.glb') 
                  for scene  in os.listdir(SCENE_DIR)
                  if os.path.isdir(os.path.join(SCENE_DIR, scene))]
scene_glb_list = np.random.choice(scene_glb_list, 
                                  size=NUM_SCENES_TO_USE, 
                                  replace=NUM_SCENES_TO_USE>len(scene_glb_list))

# Collect list of object folders in directory
# TODO: FIX THIS. Code currently assumes all objects are of the same class.sample_count
OBJECT_CLASS = 7  
# OBJECTS_DIR = joinpath(DOWNLOADED_DATA_DIR, 'objects/office_chair_models-20230215T191917Z-001')
print(f'NOTICE: all objects are assumed to be of the same class: {OBJECT_CLASS}')
# all_obj_folders = [str(obj_f) for obj_f in obj_folders_generator(OBJECTS_DIR)]
all_obj_folders = [next(obj_files_generator(obj_f)) 
                   for obj_f in 
                   get_target_folders_in_path(OBJECTS_DIR, args['objects']['chairs'])]

scene_obj_folders = [[all_obj_folders[i]] for i in 
                     np.random.randint(0, len(all_obj_folders), len(scene_glb_list))]

add_metadata_to_object_dir(OBJECTS_DIR)

# Randomize camera height
camera_height_list = np.random.uniform(0.5, 1.5, len(scene_glb_list))

# randomize pitch to fixate on the same point n meters away
distance_to_fixate_on = 4
camera_pitch_list = -np.arctan2(camera_height_list, distance_to_fixate_on)
camera_pitch_list += np.random.uniform(-0.1, 0.1, len(scene_glb_list))



for scene_idx in range(len(scene_glb_list)):
    print(f'Generating data for scene {scene_idx+1} out of {len(scene_glb_list)}')
    scene_glb = scene_glb_list[scene_idx]
    object_folders = scene_obj_folders[scene_idx]
    camera_height = camera_height_list[scene_idx]
    camera_pitch = camera_pitch_list[scene_idx]

    # Generate config
    cfg = make_cfg({
        "gpu_device_id": 1,
        
        # TODO: make 640 x 512 like spot
        "width": 640,
        "height": 480,
        
        "scene":  scene_glb,
        "scene_dataset_config": scene_config,
            
        'color_sensor_hfov_1st_person': 60.2,
        'semantic_sensor_hfov_1st_person': 60.2,
        'depth_sensor_hfov_1st_person': 60.2, # 55.9,

        "default_agent": 0,
        "sensor_height": camera_height,
        "sensor_pitch": camera_pitch,
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": False,  # RGB sensor 3rd person
        "depth_sensor_1st_person": True,  # Depth sensor
        "semantic_sensor_1st_person": True,  # Semantic sensor
        "seed": 1,
        "enable_physics": True,  # enable dynamics simulation
    })

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

    # Load all objects of interst
    for obj_idx, obj_f in enumerate(object_folders): 
        obj_attr_mgr.load_configs(obj_f)
        f_handle = obj_attr_mgr.get_file_template_handles(obj_f)[0]
        obj = rigid_obj_mgr.add_object_by_template_handle(f_handle)
        obj.semantic_id = obj_idx + 1
        obj.user_attributes.set('class', OBJECT_CLASS)

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
    if GEN_METHOD == 'Teleport':

        # Instantiate dataloader
        sim_dataloader_config = {
            # step config
            'step_type': 'teleport',
            '_m_per_px': 0.1, 

            # distance of object form obstacles
            '_min_obj_clearence_m': 0.7,    
            
            # distance of robot form obstacles
            '_min_rob_clearence_m': 0.2,   # NOTE: Only first floor for now
            '_min_obj2rob_dist_m': 1.5  ,
            '_max_obj2rob_dist_m': 4,
            '_min_path_dist_from_obstacles': 0.2,

            # if set only samples that many pose pairs
            'sample_count': 150,              
            
            # simulate gravity for this many seconds
            # NOTE: this slows the sim down a lot...
            # 'simulate_gravity_for_s': 1,  
        }
        sim_dataloader_config.update(common_dataloader_configs)


    # Setup to generate data by moving the objects
    elif GEN_METHOD == 'Kinematic':
        
        print('#'*50, "\nKinematic data generation is not fully setup yet in this script.\n",'#'*50)
        pose_gen_config = {
            '_m_per_px': 0.1, 
            '_min_obj_clearence_m': 0.5,   # distance of object form obstacles 
            '_min_rob_clearence_m': 0.2,   # distance of robot form obstacles   
                                           # NOTE: Only first floor for now
            '_min_obj2rob_dist_m': 1.5  ,
            '_max_obj2rob_dist_m': 4  ,
            '_min_path_dist_from_obstacles': 0.2,
            '_sample_count': 150,
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

        # TODO: use this for supporting more than one object... later. 
        # for h in rigid_obj_mgr.get_object_handles():
        #     obj = rigid_obj_mgr.get_object_by_handle(h)
        #     position_noise = np.random.normal(0, 0.1, 3)
        #     # Set object initial state 
        #     set_object_state_relative_to_agent(sim, obj, 
        #                     offset=np.array([0.0, 0.5, -5]) + position_noise,
        #                     orientation= mn.Quaternion.rotation(mn.Deg(0), mn.Vector3([1,0,0])),
        #                     absolute_orientation=False)
        #     # Set object motion using velocity control
        #     obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        #     vel_control = obj.velocity_control
        #     vel_control.controlling_ang_vel = True
        #     vel_control.angular_velocity = mn.Vector3((0.0, 0.2, 0.0))
        #     # TODO: set linear velocity
        #     # vel_control.controlling_lin_vel = True

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
        scene_name = scene_glb.split('/')[-2]
        save_to_sub_folder = os.path.join(SAVE_TO_FOLDER, scene_name)
        dataloader.save(save_to_sub_folder, override=OVERWRITE_EXISTING_DATA)
        print("\nData Generated Successfully!")
    except StopIteration as e:  # TODO: Make custom exception
        print("Failed to generate and save data...")
        print(e)
    except RuntimeError as e:  # TODO: Make custom exception
        print("Failed to generate and save data...")
        print(e)
    sim.close()
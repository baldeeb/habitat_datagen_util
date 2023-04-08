import numpy as np
import magnum as mn

# Set an object transform relative to the agent state
def set_object_state_relative_to_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
    absolute_orientation=True
):
    agent_node = sim.agents[0].scene_node
    agent_transform = agent_node.transformation_matrix()
    obj.translation = agent_transform.transform_point(offset)
    
    obj.rotation = orientation if absolute_orientation \
                    else orientation * agent_node.rotation

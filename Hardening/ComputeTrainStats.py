import torch
import numpy as np
from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Environment
from map_tool_box.modules import Data_Map
from map_tool_box.modules import Control
from map_tool_box.modules import Spawner
from map_tool_box.modules import Astar
from map_tool_box.modules import Utils
from map_tool_box.modules import Model
from map_tool_box.modules import Action
from map_tool_box.modules import Actor
from IPython.display import HTML
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os

def setup_on_train_inference_NH():
	map_name = 'AirSimNH'
	action_magnitudes = [1, 2, 4, 8, 16, 32]
	n_history = 3 # attention mechanism that keeps track of this many previous observations
	goal_tolerance = 4 # maximum distance (meters) required to reach goal within
	steps_multiplier = 4 # multiplies by a path's optimal number of steps to determine maximum number of steps
	horizon = 255 # farthest perceivable distance (meters) -- used for normalization
	roof_name = 'default'
	depth_sensor_name = 'DepthV1'
	memory_saver = False
	cache_size = 8

	# read any args to override above variables
	if len(sys.argv) > 1:
		arguments = Utils.parse_arguments([f'map_name:{map_name}', f'depth_sensor_name:DepthV1'])
		locals().update(arguments)

	# enviornment map -- grid data map to collect observations and detect collisions / out of bounds
	from map_tool_box.modules import Data_Map
	data_map = Data_Map.DataMapRoof(map_name, roof_name=roof_name, memory_saver=memory_saver, cache_size=cache_size)

	# action space -- only move forward in current direction, can rotate yaw to change direction
	from map_tool_box.modules import Action
	from map_tool_box.modules import Actor
	actions = []
	actions.append(Action.RotateClockwise())
	actions.append(Action.RotateCounter())
	# # forward motion
	# for magnitude in action_magnitudes:
	#     actions.append(Action.Forward(magnitude))
	# # dpad motion
	for magnitude in action_magnitudes:
		actions.append(Action.StrafeRight(magnitude))
	for magnitude in action_magnitudes:
		actions.append(Action.StrafeLeft(magnitude))
	for magnitude in action_magnitudes:
		actions.append(Action.Forward(magnitude))
	actor = Actor.Discrete(actions)

	# observation space -- n-many forward facing depth maps
	from map_tool_box.modules import Data_Transformation
	from map_tool_box.modules import Observer
	from map_tool_box.modules import Sensor
	img_observer = Observer.Observer([
			#Sensor.DataMapSensor(data_map, depth_sensor_name, Data_Transformation.Normalize(max_input=horizon)), # with normalization
			Sensor.DataMapSensor(data_map, depth_sensor_name), # no normalization
		], n_history=n_history, data_type=np.uint8)
	vec_observer = Observer.Observer([
			#Sensor.GoalDisplacement(Data_Transformation.Normalize(min_input=-1*horizon, max_input=horizon)), # with normalization
			Sensor.RelativeGoal(), # self normalizes since r, theta are different scales
			Sensor.DistanceBounds(data_map, Data_Transformation.Normalize(max_input=horizon)), # with normalization
			Sensor.CurrentDirection(Data_Transformation.Normalize(max_input=3)), # with normalization
			#Sensor.DistanceBounds(), # no normalization
			#Sensor.GoalDisplacement(), # no normalization
			#Sensor.CurrentDirection(), # no normalization
		], n_history=n_history)
	observer_dict = {
		'vec':vec_observer,
		'img':img_observer,
	}
	observer = Observer.DictObserver(observer_dict)


	# termination criteria (episodic) -- how does an episode end?
	from map_tool_box.modules import Terminator
	terminators = [
		Terminator.Goal(goal_tolerance),
		Terminator.MaxSteps(steps_multiplier),
	]
	return data_map, actor, observer, terminators



def set_hooks(train_model, prefix='', train_thrs=None):
    hook_handles = []
    thrs = {}
    stats = {} if train_thrs is not None else None
    model = train_model.sb3model.q_net

    def _register(m, name_prefix=''):
        for name, layer in m.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                handle = layer.register_forward_hook(get_hook(full_name, thrs, train_thrs, stats))
                hook_handles.append(handle)

            _register(layer, full_name)

    _register(model, prefix)
    return thrs, stats, hook_handles


def get_hook(name, thrs={}, train_thrs=None, stats=None):

	def hook(module, input, output):
		max_val = output.detach().max().item()

		# Se train_thrs è presente, aggiorna stats
		if train_thrs is not None and stats is not None and name in train_thrs:
			if name not in stats:
				stats[name] = {
					'count_superamenti': 0,
					'offsets': [],
					'media_offset': 0.0,
					'std_offset': 0.0
				}

			offset = max_val - train_thrs[name]
			stats[name]['offsets'].append(offset)

			if max_val > train_thrs[name]:
				stats[name]['count_superamenti'] += 1

			# Aggiorna media e std
			offsets_array = np.array(stats[name]['offsets'])
			stats[name]['media_offset'] = float(offsets_array.mean())
			stats[name]['std_offset'] = float(offsets_array.std())
		else:
			if name not in thrs:
				thrs[name] = max_val
			else:
				thrs[name] = max(thrs[name], max_val)

	return hook


def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()

def inference(model, map_name, model_name):

	if map_name == 'AirSimNH' and model_name == 'DRL_beta':
		data_map, actor, observer, terminators=setup_on_train_inference_NH()
	else:
		raise NotImplementedError("Inference not implemented for this map and model combination")

	n_eval_paths = 5 
	difficulties=None
	astar_version = 'version_1'
	set_name = 'train' # train val test
	others = []

	train_paths = Astar.read_curriculum(map_name, astar_version, set_name, n_eval_paths, difficulties=difficulties)
	train_spawner = Spawner.CurricululmEval(train_paths)

	environment = Environment.Episodic(data_map, train_spawner, actor, observer, terminators, others)
	
	accuracy, episodes = Control.eval(environment, model, write_path=None, save_observations=False)

def energy_from_buffer(buffer):

    buf = np.asarray(buffer).reshape(4,3)
    p0 = buf[2]
    p1 = buf[1]
    p2 = buf[0]
    dp1 = p1 - p0
    dp2 = p2 - p1

    E = np.dot(dp1, dp1) + np.dot(dp2, dp2)
    return E
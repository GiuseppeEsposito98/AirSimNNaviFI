import torch
import numpy as np
import os
# import matplotlib.pyplot as plt
# import seaborn as sns

def setup_inference_on_train():
    # header to set file paths
	import os
	cwd = os.getcwd()
	parts = cwd.split('/')
	home_dir, local_dir, dropbox_dir = None, None, None
	for i in range(len(parts)):
		if parts[i] == 'experimental':
			root_dir = '/'.join(parts[:i+1]) + '/'
		if parts[i] == 'Dropbox':
			local_dir = '/'.join(parts[:i]) + '/local/'
			dropbox_dir = '/'.join(parts[:i+1]) + '/local/'
	os.chdir(root_dir)
	import sys
	sys.path.append(root_dir)
	import utils.global_methods as gm
	gm.set_global('root_dir', root_dir)
	gm.set_global('local_dir', local_dir)
	gm.set_global('dropbox_dir', dropbox_dir)

	import map_data.map_methods as mm

	# local imports
	from configuration import Configuration
	initial_locals = locals().copy() # will exclude these parameters from config parameters written to file

	# params set from arguments passed in python call
	map_name = 'AirSimNH' # airsim map -- currently only supported is AirSimNH
	model_dir = 'models/navislim_release/v0/' # directory to model which has json configuration and sb3 pytorch neural networks 
	config_path = f'{model_dir}configuration.json' # set with args, file path to configuration.json to load train components and parameters
	model_path = f'{model_dir}model.zip' # set with args, file path to model.zip to load sb3 actor model to evaluate
	output_dir = f'{model_dir}evaluations' # set with args, directory path to folder to write results
	split_name = 'train' # set with args, i.e. train, val, test -- which portion of paths_split to use from astar path
	astar_multiplier = 4 # max time constraint to kill a path if not reached gol in time, x times the number of steps taken by the Astar ground truth path
	rooftops_version = 'v1' # keep as v1 for now
	astar_version = 'v4' # keep as v4 for now
	motion = '2d' # 2d horizontal motion, 3d is work in progress
	region = 'all' # keep as all for now
	levels_dir = f'map_data/astar_paths/{astar_version}/{map_name}/{motion}/{region}/curriculum_levels/' # directory to curriculum level directory
	device = 'cuda:0' # device to load pytorch model on
	min_level, max_level = 1, 36 # range of curric difficulty levels to test on
	max_paths_per_level = 50 # sample 100 paths from each level -- increase to do more evaluations
	job_name = None # optional param for tracking outside of this program (keep as None)
	random_seed = 777
	debug = False
	null_val = 0
	sensor_name = None
	extra_out_name = ''

	# read params from command line
	arguments = None
	# if len(sys.argv) > 1:
	# 	arguments = gm.parse_arguments(sys.argv[1:])
	# 	locals().update(arguments)
	gm.set_global('job_name', job_name)
	gm.set_global('random_seed', random_seed)
	gm.set_global('home_dir', home_dir)
	gm.set_global('local_dir', local_dir)

	assert os.path.exists(config_path), f'config_path DNE at {config_path}'
	assert os.path.exists(model_path), f'model_path DNE at {model_path}'
	assert output_dir!='null', f'output_dir not passed as arg'

	# only load Astar paths from given split name
	use_splits = [split_name]

	# setup for run, set system vars and prepare file system ##########
	gm.setup_output_dir(f'{output_dir}{extra_out_name}')

	datamap = mm.DataMap(map_name, rooftops_version)
	gm.set_global('datamap', datamap)

	# all variables here will be added to configuration parameters for reading later
	all_local_vars = locals()
	user_local_vars = {k:v for k, v in all_local_vars.items() if (not k.startswith('__') and k not in initial_locals and k not in ['initial_locals','all_local_vars', 'datamap'])}
	config_params = user_local_vars.copy() # will include all of the above parameters to config parameters written to file
	print('running job with params', config_params)

	## read old CONFIGURATION 
	# SET META DATA (anything you want here for notes, just writes to config file as a dict)
	meta = {}
	change_params = { # change parameters in components to desired value
		'device':device, # load model onto specificed pytorch device
		'read_model_path':model_path, # specify where to load model
		'debug':debug,
		'start_level':min_level,
		'min_level':min_level,
		'max_level':max_level,
		'astar_multiplier':astar_multiplier,
		}
	if arguments is not None:
		change_params.update(arguments) # add any arguments as overrides to the original navigation model
	if levels_dir != 'null':
		change_params['levels_dir']  = levels_dir
	if sensor_name is not None:
		change_params['sensor_name'] = sensor_name
	change_params['use_splits'] = use_splits
	configuration = Configuration.load(
		config_path, # read all components in this config file
		read_modifiers=False, # do not load modifiers used in train configuration - we will make new ones
		skip_components = [ # do not load these components because we will overwrite them for testing
			'Curriculum',
			'Saver',
			],
		change_params=change_params
		)
	configuration.update_meta(meta)
	model = configuration.get_component('Model')
	model._load_arguments = {'buffer_size':0}

	# parameters set from config file
	motion = configuration.get_parameter('motion')
	astar_version = configuration.get_parameter('astar_version')
	region = configuration.get_parameter('region')
	null_val = configuration.get_parameter('null_val')
	gm.set_global('null_val', null_val)
	configuration.set_parameter('random_seed', random_seed)

	## CONTROLLER -- we will test on config
	from NaviAPPFI.Controller.test_wo_sim_41293c68fe7e15560d26ba8fa6c1bf377a7df4fd import Test
	controller = Test(
			environment_component = 'EnvironmentVal', # environment to run test in
			model_component = 'Model', # used to make predictions
			spawner_component = 'Spawner', # used to make predictions
			results_path = f'{output_dir}{extra_out_name}evaluation__{split_name}.p',
			job_name = job_name,
		)
	configuration.set_controller(controller)
	print(f'{output_dir}{extra_out_name}evaluation__{split_name}.json')

	# CONNECT COMPONENTS
	configuration.connect_all()
	controller._spawner.set_temp_n_paths(max_paths_per_level)
	controller._spawner.random_splits[split_name] = False
	controller._spawner.current_splits[split_name] = False
	controller._spawner.set_active_split(split_name)
	controller._spawner.reset_learning()
	return configuration, output_dir

def set_hooks(train_configuration, prefix='', output_dir='', train_thrs=None):
    hook_handles = []
    thrs = {}
    stats = {} if train_thrs is not None else None
    model = train_configuration.controller._model._sb3model.policy.q_net

    def _register(m, name_prefix=''):
        for name, layer in m.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            log_folder = os.path.join(output_dir, name)
            if not os.path.exists(log_folder):
                os.mkdir(log_folder)

            if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                handle = layer.register_forward_hook(get_hook(full_name, thrs, log_folder, train_thrs, stats))
                hook_handles.append(handle)

            _register(layer, full_name)

    _register(model, prefix)
    return thrs, stats, hook_handles


def get_hook(name, thrs={}, log_folder='', train_thrs=None, stats=None):

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

# def set_hooks(train_configuration, prefix='', output_dir=''):
#     hook_handles = []
#     thrs = {}
#     model = train_configuration.controller._model._sb3model.policy.q_net
#     def _register(m, name_prefix=''):
#         for name, layer in m.named_children():
#             full_name = f"{name_prefix}.{name}" if name_prefix else name
#             log_folder = os.path.join(output_dir, name)
#             if not os.path.exists(log_folder):
#                  os.mkdir(log_folder)
#             if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
#                 handle = layer.register_forward_hook(get_hook(full_name, thrs, output_dir))
#                 hook_handles.append(handle)

#             # Ricorsivamente continua con i figli
#             _register(layer, full_name)

#     _register(model, prefix)

#     return thrs, hook_handles


# def get_hook(name, thrs={}, log_folder=''):
#     def hook(module, input, output):
#         # Get max from output tensor (regardless of shape)
#         max_val = output.detach().max().item()
                  
#         if name not in thrs:
#             thrs[name] = max_val
#         else:
#             thrs[name] = max(thrs[name], max_val)
#     return hook

def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()

def inference(train_configuration, req_stat = None):

    # RUN CONTROLLER
    res = train_configuration.controller.run()
	# for el in res:
	# 	print('el')
	# 	print(el)

    # done
    print('Evaluations done!')
    train_configuration.disconnect_all()
    return res[1]

def energy_from_buffer(buffer):

    buf = np.asarray(buffer).reshape(4,3)
    p0 = buf[2]
    p1 = buf[1]
    p2 = buf[0]
    dp1 = p1 - p0
    dp2 = p2 - p1

    E = np.dot(dp1, dp1) + np.dot(dp2, dp2)
    return E
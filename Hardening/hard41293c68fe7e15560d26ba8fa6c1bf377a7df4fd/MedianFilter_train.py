
# header to set file paths
import os
cwd = os.getcwd()
parts = cwd.split('/')
root_dir, local_dir, dropbox_dir = None, None, None
for i in range(len(parts)):
	if parts[i] == 'experimental':
		root_dir = '/'.join(parts[:i+1]) + '/'
		local_dir = root_dir
	if parts[i] == 'Dropbox':
		local_dir = '/'.join(parts[:i]) + '/local/'
		dropbox_dir = '/'.join(parts[:i+1]) + '/'
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
import utils.global_methods as gm
gm.set_global('root_dir', root_dir)
gm.set_global('local_dir', local_dir)
gm.set_global('dropbox_dir', dropbox_dir)
print('ROOT', root_dir)

# optional imports of useful global methods
import utils.global_methods as gm # common utility methods
import map_data.map_methods as mm # data fetching and handling methods
import reinforcement_learning.reinforcement_methods as rm # common overarching methods used for DRL
from supervised_learning import supervised_methods as sm

## local imports
from configuration import Configuration
import random
import numpy as np








# ****** CUSTOM LAYERS MUST BE DEFINED HERE FOR STABLE-BASELINES3 TO GET DROPOUT TO WORK

import torch as th
import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from supervised_learning import supervised_methods as sm
import copy 
from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.MedianFilter import MedianPool


def str_to_class(class_str):
    if class_str == 'CombinedExtractor_tim':
        return CombinedExtractor_tim
    if class_str == 'NatureCNN_tim':
        return NatureCNN_tim
    if class_str == 'CustomCNN':
        return CustomCNN
    if class_str == 'CustomCombinedExtractor':
        return CustomCombinedExtractor
    if class_str == 'CustomQNetwork':
        return CustomQNetwork
    if class_str == 'CustomDQNPolicy':
        return CustomDQNPolicy

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            MedianPool(3),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(MedianPool(3), nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        scale = 1
        for key, subspace in observation_space.spaces.items():
            if key == "img":
                dropout_rate = 0.2
                n_input_channels = subspace.shape[0]
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
					nn.Conv2d(n_input_channels, int(32*scale), kernel_size=8, stride=4, padding=0),
					nn.ReLU(),
					nn.Dropout(p=dropout_rate),
                    MedianPool(3),
					nn.Conv2d(int(32*scale), int(64*scale), kernel_size=4, stride=2, padding=0),
					nn.ReLU(),
					nn.Dropout(p=dropout_rate),
                    MedianPool(3),
					nn.Conv2d(int(64*scale), int(64*scale), kernel_size=3, stride=1, padding=0),
					nn.ReLU(),
					nn.Dropout(p=dropout_rate),
					nn.Flatten(),
				)
                total_concat_size += features_dim
            elif key == "vec":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(MedianPool(3), nn.Linear(subspace.shape[0], 16))
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

class CombinedExtractor_tim(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        tim_cnn_class = NatureCNN,
        tim_cnn_kwargs = None,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        if isinstance(tim_cnn_class, str):
            tim_cnn_class = str_to_class(tim_cnn_class)

        extractors: dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                if tim_cnn_kwargs is None:
                    extractors[key] = tim_cnn_class(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                else:
                    extractors[key] = tim_cnn_class(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image, **tim_cnn_kwargs)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

# this is modified from sb3 to change the size
class NatureCNN_tim(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        scale:float = 1,
        dropout_rate=0,
        dropout_scale=True,
    ) -> None:
        assert isinstance(observation_space, gym.spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        if dropout_rate > 0:
            if dropout_scale:
                scale = scale * 1/(1-dropout_rate)
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, int(32*scale), kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                MedianPool(3),
                nn.Conv2d(int(32*scale), int(64*scale), kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                MedianPool(3),
                nn.Conv2d(int(64*scale), int(64*scale), kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, int(32*scale), kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                MedianPool(3),
                nn.Conv2d(int(32*scale), int(64*scale), kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                MedianPool(3),
                nn.Conv2d(int(64*scale), int(64*scale), kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(MedianPool(3), nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
import copy
from typing import Any, Optional

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    #CombinedExtractor,
    FlattenExtractor,
    #NatureCNN,
    #create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
class CustomQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        mlp_kwargs: dict,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.mlp_kwargs = copy.deepcopy(mlp_kwargs)
        action_dim = int(self.action_space.n)  # number of actions
        self.mlp_kwargs['layers'] = [features_dim] + self.mlp_kwargs['layers'] + [action_dim]
        self.q_net = sm.create_mlp(**self.mlp_kwargs)

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                mlp_kwargs=self.mlp_kwargs,
                features_dim=self.features_dim,
                features_extractor=self.features_extractor,
            )
        )
        return data


class CustomDQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: CustomQNetwork
    q_net_target: CustomQNetwork

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        mlp_kwargs: dict = {},
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        self.mlp_kwargs = mlp_kwargs

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "mlp_kwargs": self.mlp_kwargs,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> CustomQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return CustomQNetwork(**net_args).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                mlp_kwargs=self.net_args["mlp_kwargs"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode

# ****** END OF CUSTOM LAYERS










initial_locals = locals().copy() # will exclude these parameters from config parameters written to file


## default parameters

# rewards
# scale_factor = 1/10
# StepReward = -0.446
# GoalDistanceReward = -0.333*scale_factor
# CollisionReward = -0.120
# BoundsReward = -0.120
# TurnReward = -0.153
# #ClearanceReward = 0.671*scale_factor
# GoalReward = 16.081
# MaxStepsReward = 0

StepReward = -1
GoalDistanceReward = -1
CollisionReward = 0
BoundsReward = 0
TurnReward = -0.1
GoalReward = 40
MaxStepsReward = -10

# misc
random_seed = 777
job_name = f'null_{random.randint(0, 1_000_000)}'
continue_training = True
continue_model_at = '/leonardo_work/IscrC_HardNet/MF/modeling/model_checkpoint.zip' # specify model path to continue training from
continue_config_at = '/leonardo_work/IscrC_HardNet/MF/modeling/configuration_model_checkpoint.json' # specify configuration path to continue training from
continue_buffer_at = None
overwrite_directory = False # True will erase all files at otuput working directory
version = ''
null_val = 0

# environment
map_name = 'AirSimNH'
rooftops_version = 'v1'
region = 'all'
map_resolution_x, map_resolution_y, map_resolution_z = 2, 2, 4
stop_at_invalid = True
add_timers = False

# levels spawner
astar_name = 'all'
astar_version = 'v3' 
level_proba = 0.7
split_name = 'train'

# action space
motion = '2d'
translation_speed = 4 # meters/sec
rotation_speed = 30 # degrees/sec
if astar_version in ['v1']:
	action_weights = {
		'rotate_right':90/rotation_speed,
		'rotate_left':90/rotation_speed,
		'move_forward_2':2/translation_speed,
		'move_forward_4':4/translation_speed,
		'move_forward_8':8/translation_speed,
		'move_forward_16':16/translation_speed,
		'move_forward_32':32/translation_speed,
	}
if astar_version in ['v2', 'v3', 'v4']:
	action_weights = {
		'rotate_right':90/rotation_speed,
		'rotate_left':90/rotation_speed,
		'move_right_2':2/translation_speed,
		'move_right_4':4/translation_speed,
		'move_right_8':8/translation_speed,
		'move_right_16':16/translation_speed,
		'move_right_32':32/translation_speed,
		'move_left_2':2/translation_speed,
		'move_left_4':4/translation_speed,
		'move_left_8':8/translation_speed,
		'move_left_16':16/translation_speed,
		'move_left_32':32/translation_speed,
		'move_forward_2':2/translation_speed,
		'move_forward_4':4/translation_speed,
		'move_forward_8':8/translation_speed,
		'move_forward_16':16/translation_speed,
		'move_forward_32':32/translation_speed,
	}
if motion == '3d':
	action_weights['move_down_4'] = 4/translation_speed
	action_weights['move_up_4'] = 4/translation_speed
	
# observation space
nPast = 3
sensor_name = 'DepthV1'
id_name = 'alpha' # when reading in observation data, which ID key words to use
vector_sensors = {
	'RelativeGoal':True,
	'DistanceBounds':True,
	'DroneDirection':True,
}
vector_sensor_n = {
	'RelativeGoal':2,
	'DistanceBounds':1,
	'DroneDirection':1,
}
vector_length_remember = 0
vector_length_forget = 0
for vector_sensor_name in vector_sensors:
	if vector_sensors[vector_sensor_name]:
		vector_length_remember += vector_sensor_n[vector_sensor_name]
	else:
		vector_length_forget += vector_sensor_n[vector_sensor_name]

# reward function
compute_time = 0.14 # seconds
goal_tolerance = 4
astar_multiplier = 4 # determines max length of an episode
include_action_weights = True # includes in maxRrewards

# curriculum learning
level_freq = 10_000
eval_freq = 100_000
ckpt_freq = 100_000
max_turns = 0 # used to load curric levels with number of turns in gt path up to this many 
use_custom_exp = False
init_exp_rate = 1.0
final_exp_rate = 0.05
decay_exp_rate = 0.00005 # decays exploration rate by this much every episode, so make this a small value
eval_on_levelup = True
max_episodes = 2_000_000

# model
model_type = 'dqn'

# DQN policy
gamma = 0.99
net_arch_nodes = 64
net_arch_layers = 3
feature_extractor_scale = 1
feature_extractor_dim = 256
total_policy_scale = 2
device = 'cuda:0'
min_level, max_level, start_level = 23, 46, 23 # index range of path difficulties to train and evaluate on, inclusive
output_dir = '/leonardo_work/IscrC_HardNet/MF/'
dropout_rate = 0
dropout_scale = True

# learning algorithm
total_timesteps = 1_000_000_000 # maximum number of timesteps to train on
	# SB3 default is 1e6, Microsoft uses 5e5
buffer_size = 20 # number of recent steps (observation, action, reward) tuples to store in memory to train on -- takes up memory
	# ^^ SB3 default is 1e6, Microsoft uses 5e5, I typically have to use less because of memory constraints
#exploration_fraction =  0.1
stop_annealing = 100_000 # number of steps to stop annealing the exploration rate at
	
# will step through each env.step() and display() and prompt user for input
debug = False

# read params from command line
if len(sys.argv) > 1:
	arguments = gm.parse_arguments(sys.argv[1:])
	locals().update(arguments)

# jupyter notebook params
#output_dir = 'map_data/imitation/V2/' # max level 0, baseline time rewards
#debug = True

assert output_dir!='null', f'output_dir not passed as arg'
gm.set_global('job_name', job_name)
gm.set_global('root_dir', root_dir)
gm.set_global('local_dir', local_dir)
gm.set_global('device', device)
gm.set_global('version', version)
gm.set_global('null_val', null_val)
gm.set_random_seed(random_seed)
gm.set_global('str_to_class', str_to_class)

# set rewards based on params
rewards = {}
rewards['StepReward'] = StepReward
rewards['GoalDistanceReward'] = GoalDistanceReward
if CollisionReward != 0:
	rewards['CollisionReward'] = CollisionReward
if BoundsReward != 0:
	rewards['BoundsReward'] = BoundsReward
rewards['TurnReward'] = TurnReward
#rewards['ClearanceReward'] = ClearanceReward
rewards['GoalReward'] = GoalReward
rewards['MaxStepsReward'] = MaxStepsReward

# set variable subpaths from root directories and params set above
astar_dir = f'{root_dir}map_data/astar_paths/'
observations_dir = f'{root_dir}map_data/observations/'
levels_dir = f'map_data/astar_paths/{astar_version}/{map_name}/{motion}/{region}/curriculum_levels/' # directory to curriculum level directory
#clearances_path = f'{root_dir}map_data/clearance/{map_name}/{motion}/{rooftops_version}/clearances.p'
complete_path = f'{output_dir}completed.p' # path to check if this job is done already
gm.set_global('complete_path', complete_path)

# how to handle if completed path already exists (showing a previous job has finished this data collection already)
# if os.path.exists(complete_path):
# 	if overwrite_directory:
# 		os.remove(complete_path)
# 	else:
# 		gm.progress(job_name, 'complete')
# 		sys.exit()

# setup output directory
gm.setup_output_dir(output_dir, overwrite_directory)
os.makedirs(output_dir+'modeling', exist_ok=True)

# bounds drone can move in
datamap = mm.DataMap(map_name, rooftops_version)
gm.set_global('datamap', datamap)
x_bounds, y_bounds, z_bounds = datamap.get_bounds(region, motion)
x_vals = [x for x in range(x_bounds[0], x_bounds[1]+1, map_resolution_x)]
y_vals = [y for y in range(y_bounds[0], y_bounds[1]+1, map_resolution_y)]
z_vals = [z for z in range(z_bounds[0], z_bounds[1]+1, map_resolution_z)]
d_vals = [0, 1, 2, 3] # what directions are accessible by drone
gm.set_global('x_bounds', x_bounds)
gm.set_global('y_bounds', y_bounds)
gm.set_global('z_bounds', z_bounds)
gm.set_global('d_vals', d_vals)
	
data_vector_sensors = {}
data_image_sensors = {}
if sensor_name == 'None':
	pass
#elif 'f' in sensor_name:
#	data_vector_sensors = {sensor_name:True}
else:
	data_image_sensors = {sensor_name:True}

# misc
exploration_fraction = stop_annealing / total_timesteps
	# SB3 default is 0.1*total_timesteps, Microsoft uses 5e4

# all variables here will be added to configuration parameters for reading later
all_local_vars = locals()
user_local_vars = {k:v for k, v in all_local_vars.items() if (not k.startswith('__') and k not in initial_locals and k not in ['initial_locals','all_local_vars', 'datamap'])}
config_params = user_local_vars.copy() # will include all of the above parameters to config parameters written to file
print('running job with params', config_params)

# COMPONENTS

# continue training will load runs folder and pick up where it left off
# load configuration file and create object to save and connect components
loaded_from_file = False
change_params={ # change parameters in components to desired value
	'device':device, # load model onto specificed pytorch device
	'max_episodes':max_episodes,
	'total_timesteps':total_timesteps,
	'max_level':max_level,
	'output_dir':output_dir,
}
if continue_training and continue_config_at is None:
    continue_config_at = f'{output_dir}modeling/configuration_model_checkpoint.json'
if continue_training and continue_model_at is None:
    continue_model_at = f'{output_dir}modeling/model_checkpoint.zip'
if continue_training:
	loaded_from_file = True
	configuration = Configuration.load(continue_config_at, change_params=change_params)
	# make controller to run configuration on (we will train a model)
	from controllers.train import Train
	controller = Train(
		model_component = 'Model',
		environment_component = 'Environment',
		continue_training = continue_training,
		total_timesteps = total_timesteps,
		name = 'controller',
		)
	configuration.set_controller(controller)
	for key in config_params:
		new_param = config_params[key]
		if key in change_params:
			new_param = change_params[key]
		configuration.set_parameter(key, new_param)
	meta = configuration.meta
	meta['continued_training'] = True
	# read model and/or replay buffer
	# get highest level complete=
	model_component = configuration.get_component('Model')
	model_component.read_model_path = continue_model_at
	model_component.read_replay_buffer_path = continue_buffer_at
	gm.speak('continuing training...')

# if not continuing training then make a brand spaking new config
else:
	# set meta data (anything you want here, just writes to config file as a dict)
	meta = {
		}

	## make a new configuration file to add components to 
		# this obj will be used to serialize components, and log experiment history
		# any components created after making this configuration file will auto be added to it
		# components use the name of other components which will be created and connected later
		# this is done to handle different objects which need to be connected to eachother and connected in different orders
		# there is a baked in priority que for connecting components if you need to handle this differently
	configuration = Configuration(meta, add_timers=add_timers)
	# make controller to run configuration on (we will train a model)
	from controllers.train import Train
	controller = Train(
		model_component = 'Model',
		environment_component = 'Environment',
		continue_training = continue_training,
		total_timesteps = total_timesteps,
		name = 'controller',
		)
	configuration.set_controller(controller)
	for key in config_params:
		configuration.set_parameter(key, config_params[key])

	## create environment component to handle step() and reset() for DRL model training
	from environments.goalenv import GoalEnv
	goal_env = GoalEnv(
		drone_component = 'Drone', 
		actor_component = 'Actor', 
		observer_component = 'Observer', 
		rewarder_component = 'Rewarder',
		model_component = 'Model',
		map_component = 'Map',
		spawner_component = 'Spawner',
		crash_handler = False,
		debug = debug,
		name = 'Environment',
		)

	# create map object
	from maps.datamap import DataMap
	DataMap(
		name = 'Map',
		x_bounds = x_bounds,
		y_bounds = y_bounds,
		z_bounds = z_bounds,
		)

	# drone controller component - we will use AirSim
		# this can also be real world drone controller like Tello
	from drones.etherial import Etherial
	Etherial(
		map_component = 'Map',
		stop_at_invalid = stop_at_invalid,
		name = 'Drone',
		)

	## REWARD FUNCTION
	# constant value at each step
	if 'StepReward' in rewards:
		from rewards.step import Step
		Step(
			name = 'StepReward',
			)
	# distance to goal
	if 'GoalDistanceReward' in rewards:
		from rewards.goaldistance import GoalDistance
		GoalDistance(
			drone_component = 'Drone',
			goal_component = 'Spawner',
			name = 'GoalDistanceReward',
			)
	# if action is in given index list
	if 'TurnReward' in rewards:
		from rewards.action import Action
		Action(
			action_idxs = [0,1],
			name = 'TurnReward',
			)
	# distance to collidable objects
	# if 'ClearanceReward' in rewards:
	# 	from rewards.clearance import Clearance
	# 	Clearance(
	# 		drone_component = 'Drone',
	# 		clearances_path = clearances_path,
	# 		name = 'ClearanceReward',
	# 		)
	# heavy penalty for collision
	if 'CollisionReward' in rewards:
		from rewards.collision import Collision
		Collision(
			drone_component = 'Drone',
			name = 'CollisionReward',
			)
	# heavy penalty for out of bounds
	if 'BoundsReward' in rewards:
		from rewards.bounds import Bounds
		Bounds(
			drone_component = 'Drone',
			x_bounds = x_bounds,
			y_bounds = y_bounds,
			z_bounds = z_bounds,
			name = 'BoundsReward',
			)
	# heavy reward for reaching goal
	if 'GoalReward' in rewards:
		from rewards.goal import Goal
		Goal(
			drone_component = 'Drone',
			goal_component = 'Spawner',
			tolerance = goal_tolerance, # must reach goal within this many meters
			name = 'GoalReward',
			)
	# heavy penalty for using too many steps
	if 'MaxStepsReward' in rewards:
		from rewards.maxsteps import MaxSteps
		MaxSteps(
			spawner_component = 'Spawner',
			use_astar = True,
			astar_multiplier = astar_multiplier, # max step size is this many times the astar length
			name = 'MaxStepsReward',
			)
	# REWARDER
	from rewarders.schema import Schema
	Schema(
		rewards_components = list(rewards.keys()),
		reward_weights = [rewards[name] for name in rewards], 
		name = 'Rewarder',
		)

	## ACTION SPACE
	# we will just move forward and rotate for this example
	from actions.object import Object 
	for key_name in action_weights:
		action_params = {}
		if 'rotate' in key_name:
			action_name = key_name
		elif 'move' in key_name:
			parts = key_name.split('_')
			action_name = parts[0] + '_' + parts[1]
			magnitude = int(parts[2])
			action_params = {'magnitude':magnitude}
		print(action_name, action_params, key_name)
		Object(
			object_component = 'Drone', 
			action_name = action_name,
			action_params = action_params,
			name = key_name,
			)
	## ACTOR
	from actors.discrete import Discrete
	Discrete(
		actions_components = list(action_weights.keys()),
		name = 'Actor',
		)

	## OBSERVATION SPACE
	# TRANSFORMERS
	from transformers.normalize import Normalize
	Normalize(
		min_input = 0, # min direction
		max_input = 3, # max direction
		name = 'NormalizeDirection',
		)
	Normalize(
		min_input = 0, # 
		max_input = 255, # horizon
		name = 'NormalizeDistance',
		)
	# SENSORS
	# sense drone position
	if 'RelativeGoal' in vector_sensors:
		from sensors.relativegoal import RelativeGoal
		RelativeGoal(
			drone_component = 'Drone',
			goal_component = 'Spawner',
			prefix = 'relative_goal',
			transformers_components = [], 
			name = 'RelativeGoal',
			)
	# sense distance to bounds in front of drone
	if 'DistanceBounds' in vector_sensors:
		from sensors.distancebounds import DistanceBounds
		DistanceBounds(
			drone_component = 'Drone',
			x_bounds = x_bounds,
			y_bounds = y_bounds,
			z_bounds = z_bounds,
			include_z = True if motion in '3d' else False,
			transformers_components = [
					'NormalizeDistance',
				],
			name = 'DistanceBounds',
		)
	# sense current drone direction
	if 'DroneDirection' in vector_sensors:
		from sensors.direction import Direction
		Direction(
			drone_component = 'Drone',
			transformers_components = [
					'NormalizeDirection',
				],
			name = 'DroneDirection',
		)
	for sensor_name in data_vector_sensors:
		sensor_info = gm.read_json(f'{observations_dir}{sensor_name}/info.json')
		vector_length = sensor_info['vector_length']
		vector_sensor_n[sensor_name] = vector_length
		# single sensor
		from sensors.datamap import DataMap
		DataMap(
			drone_component = 'Drone',
			sensor_name = sensor_name,
			sensor_dir = f'{observations_dir}{sensor_name}/',
			transformers_components=[],
			name = sensor_name,
		)
		if data_vector_sensors[sensor_name]:
			vector_length_remember += vector_length
		else:
			vector_length_forget += vector_length
	if len(data_image_sensors) > 0:
		image_bands_remember = 0
		image_bands_forget = 0
		for sensor_name in data_image_sensors:
			sensor_info = gm.read_json(f'{observations_dir}{sensor_name}/info.json')
			image_bands, image_height, image_width = sensor_info['array_size']
			# single sensor
			from sensors.datamap import DataMap
			DataMap(
				drone_component = 'Drone',
				sensor_name = sensor_name,
				sensor_dir = f'{observations_dir}{sensor_name}/',
				transformers_components=[],
				name = sensor_name,
			)
			if data_image_sensors[sensor_name]:
				image_bands_remember += image_bands
			else:
				image_bands_forget += image_bands
		# OBSERVER
		# currently must count vector size of sensor output
		from observers.single import Single
		Single(
			sensors_components = [name for name in vector_sensors] + [name for name in data_vector_sensors],
			vector_length_forget = vector_length_forget,
			vector_length_remember = vector_length_remember,
			nPast = nPast,
			null_if_in_obj = True, # inside an object
			null_if_oob = True, # out of bounds
			name = 'VecObserver',
			)
		Single(
			sensors_components = [name for name in data_image_sensors], 
			is_image = True,
			image_height = image_height, 
			image_width = image_width,
			image_bands_forget = image_bands_forget,
			image_bands_remember = image_bands_remember,
			nPast = nPast,
			null_if_in_obj = True, # inside an object
			null_if_oob = True, # out of bounds
			name = 'ImgObserver',
			)
		from observers.multi import Multi
		Multi(
			vector_observer_component = 'VecObserver',
			image_observer_component = 'ImgObserver',
			name = 'Observer',
			)
		## MODEL
		from sb3models.dqn import DQN
		addtional_scale= 1
		if dropout_scale:
			addtional_scale = 1/(1-dropout_rate)
		#from custom_policies import CustomDQNPolicy
		model = DQN(
			environment_component = 'Environment',
			policy = 'CustomDQNPolicy', #'MultiInputPolicy',
			buffer_size = buffer_size,
			device = device,
			exploration_fraction = exploration_fraction,
			# calling custom methods for 3rd party libraries is messy
			# note that the _class features are strings to make them json configuration compatidble
			# I wrote a str_to_class() method in sb3model to adjust for this by converting to class at run time
			policy_kwargs = {
				'mlp_kwargs':{
            		'layers':[int(addtional_scale*total_policy_scale*net_arch_nodes) for _ in range(net_arch_layers)],
                    'dropout':[dropout_rate for _ in range(net_arch_layers+1)]},
				'features_extractor_class': 'CombinedExtractor_tim', # CombinedExtractor_tim
				'features_extractor_kwargs':{
					'cnn_output_dim':int(total_policy_scale*feature_extractor_dim),
					'normalized_image':False,
					'tim_cnn_class':'NatureCNN_tim', # NatureCNN_tim
					'tim_cnn_kwargs':{
						'scale':total_policy_scale*feature_extractor_scale,
						'dropout_rate':dropout_rate,
						'dropout_scale':dropout_scale,
					},
				},
			},
			use_custom_exp=use_custom_exp,
			init_exp_rate=init_exp_rate,
			final_exp_rate=final_exp_rate,
			decay_exp_rate=decay_exp_rate,
			name = 'Model',
		)
	else:
		# currently must count vector size of sensor output
		from observers.single import Single
		Single(
			sensors_components = [name for name in vector_sensors] + [name for name in data_vector_sensors],
			vector_length_forget = vector_length_forget,
			vector_length_remember = vector_length_remember,
			nPast = nPast,
			null_if_in_obj = True, # inside an object
			null_if_oob = True, # out of bounds
			name = 'Observer',
			)
		## MODEL
		from sb3models.dqn import DQN
		model = DQN(
			environment_component = 'Environment',
			policy = 'MlpPolicy',
			buffer_size = buffer_size,
			gamma = gamma,
			device = device,
			exploration_fraction = exploration_fraction,
			# calling custom methods for 3rd party libraries is messy
			# note that the _class features are strings to make them json configuration compatidble
			# I wrote a str_to_class() method in sb3model to adjust for this by converting to class at run time
			policy_kwargs = {
				'net_arch':[int(total_policy_scale*net_arch_nodes) for _ in range(net_arch_layers)],
			},
			use_custom_exp=use_custom_exp,
			init_exp_rate=init_exp_rate,
			final_exp_rate=final_exp_rate,
			decay_exp_rate=decay_exp_rate,
			name = 'Model',
		)

	# SPAWNER
		# moves drone to desired starting location
		# sets the target goal since it is typically dependent on the start location
	current_splits = { # True with use current level only, False will use min-max levels
				'train':False,
				'val':False,
				'test':False,
			}
	from spawners.levels import Levels			
	Levels(
		drone_component = 'Drone',
		levels_dir = levels_dir,
		min_level = min_level,
		max_level = max_level,
		start_level = start_level, # will start at this level unless specified with level argument
		split_name = 'train',
		current_splits = current_splits,
		level_proba = level_proba,
		name = 'Spawner',
	)

	## MODIFIERS
		# modifiers are like wrappers, and will add functionality before or after any component
	# CURRICULUM LEARNING
		# this modifier will be called at the end of every episode to see the percent of succesfull paths
		# if enough paths were succesfull then this will level up to harder goal
	from modifiers.quick2 import Quick2
	Quick2(
		base_component = 'Environment',
		parent_method = 'end',
		order = 'post',
		spawner_component = 'Spawner', # which component to level up
		model_component = 'Model',
		min_level = min_level, # will sample paths from min to max curric
		max_level = max_level, # can level up this many times after will terminate DRL learning loop
		start_level = start_level, # will start at this level unless specified with level argument
		eval_controller_component = 'EvalController',
		update_progress = True,
		max_episodes = max_episodes,
		level_freq = level_freq,
		eval_freq = eval_freq,
		ckpt_freq = ckpt_freq,
		name = 'Curriculum',
	)

	# val/test components
	# ENVIRONMENT
	GoalEnv(
		drone_component = 'Drone', 
		actor_component = 'Actor', 
		observer_component = 'Observer', 
		rewarder_component = 'Rewarder',
		model_component = 'Model',
		map_component = 'Map',
		spawner_component = 'Spawner',
		crash_handler = False,
		name = 'EnvironmentVal',
		)
	# SAVERS - save observations and/or states at each step
	# Saver(
	# 	base_component = 'Environment',
	# 	parent_method = 'end',
	# 	track_vars = [
	# 				#'observations', # uncomment this to write observations to file (can take alot of disk space)
	# 				#'states',
	# 				],
	# 	order = 'post',
	# 	save_config = True,
	# 	save_benchmarks = False,
	# 	frequency = 10_000,
	# 	name='Saver',
	# )
	# from modifiers.saver import Saver
	# checkpoint_freq = total_timesteps # default is to just save at end of training
	# Saver(
	# 	base_component = 'Model', # Modifiers will execute after/before this given component is used
	# 	parent_method = 'end', # Modifiers will execute after/before this given component's method is used
	# 	order = 'post', # Modifiers will run code either before (pre) or after (post) given component call
	# 	track_vars = [
	# 				'model', 
	# 				#'replay_buffer', # this can cost alot of memory so default is to not save
	# 				],
	# 	write_folder = output_dir + 'modeling/',
	# 	frequency = 1_000, # save every this many episodes
	# 	save_config = True,
	# 	save_benchmarks = False,
	# 	name = 'ModelingSaver',
	# )
	
	from others.eval import Eval
	Eval(
		environment_component = 'EnvironmentVal',
		model_component = 'Model',
		spawner_component = 'Spawner',
		name = 'EvalController'
	)

	if debug:
		goal_env._nPast = nPast
		goal_env._reward_names = rewards
		goal_env._vector_sensor_n = vector_sensor_n

# CONNECT COMPONENTS
configuration.connect_all()
gm.speak('all components connected...')

# from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.MedianFilter import implement_median_filter
# configuration = implement_median_filter(configuration)

print(configuration.controller._model._sb3model.policy.q_net)
# configuration.controller._model._sb3model.policy.q_net = configuration.controller._model._sb3model.policy.q_net.to(device)

# set default exit reason
configuration.set_parameter('termination_reason', 'max steps')

#if not loaded_from_file:
# WRITE CONFIGURATION
#configuration.save()

# WRITE CONTROLLER
controller.save(output_dir + 'train_controller.json')

# RUN CONTROLLER
gm.speak('running controller...')
configuration.controller.run()

# DISCONNECT
configuration.disconnect_all()

# done
gm.speak('training complete! evaluating...')
# curriculum = configuration.get_component('Curriculum')
# goal_env = configuration.get_component('Environment')
# if configuration.get_parameter('termination_reason') is None:
# 	configuration.set_parameter('termination_reason', 'max steps')
# curriculum.save_model('model_final')
# curriculum.full_eval()

# gm.pk_write(True, complete_path)
# gm.speak(f'complete {goal_env.episode_counter} eps {max_level}')
# gm.progress(job_name, f'complete {goal_env.episode_counter} eps {max_level} lvl')

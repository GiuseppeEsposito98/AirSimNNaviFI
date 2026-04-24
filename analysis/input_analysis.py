from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Environment
from map_tool_box.modules import Data_Map
# from map_tool_box.modules import Control
from map_tool_box.modules import Spawner
# from map_tool_box.modules import Other
from map_tool_box.modules import Astar
from map_tool_box.modules import Utils
from map_tool_box.modules import Model
from IPython.display import HTML
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os
import json
import argparse
import torch
from copy import deepcopy
from map_tool_box.AirSimNNaviFI.analysis import Control



def get_argparse():
    parser = argparse.ArgumentParser(description='DQN configuration')
    parser.add_argument('--fsim_config', help='Fault simulation configuration json file path')
    parser.add_argument('--target_layer', help='The layer that the simulation has to target')
    parser.add_argument('--trials', help='Number of FI trials to run')
    parser.add_argument('--fsim_log_name', required=True, help='Directory name where the results are stored')
    parser.add_argument('--hardening', default=None, required=False, help='Name of the hardening Technique to implement')
    parser.add_argument('--paths_number', default=5, help='Number of paths to evaluate on (if not specified will evaluate on 5 paths per difficulty level)')
    parser.add_argument('--difficulties', default=None, help='Difficulty levels as a list (if not specified will evaluate on representative difficulty levels)')
    return parser

def main(args):
        
    print(f'config_path: {config_path}')

    # read model from file
    model_path = Path(model_directory, 'model.zip')
    model = Model.read_model(model_path)

    # set parameters for which set of Astar paths to evaluate on
    astar_version = 'version_1'
    set_name = 'test' # train val test
    n_paths = deepcopy(int(args.paths_number)) # if None then will read all paths from file, otherwise an integer value specifying number of paths PER DIFFICULTY
    difficulties = deepcopy(args.difficulties) # if None then will read all difficulties from file, otherwise expects a list of difficulty keys
    fsim_config = deepcopy(args.fsim_config)
    hardening = deepcopy(args.hardening)
    target_layer = deepcopy(args.target_layer)
    trials = deepcopy(args.trials)
    # read paths from file (usese some variables read in from config.py file) 
        # -- you can overwrite map_name or astar_version, but the default values or those used to train the model
    paths = Astar.read_curriculum(map_name, astar_version, set_name, n_paths, difficulties)

    # create spawner object to iterate through paths from environment
    spawner = Spawner.CurricululmEval(paths)
        
    # add any additional components to config
    others = []

    # create environment that we will step through (uses some objects read in from config.py file)
    environment = Environment.Episodic(data_map, spawner, actor, observer, terminators, others)

    

if __name__ == "__main__":
    # point to model you wish to evaluate
    models_directory = Utils.get_global('models_directory') # check to make sure this is correct on your local computer (it should be auto)
    model_name = 'DRL_beta' # change to proper model name if needed (name is also sub-directory path)
    project_name = 'AirSim_Navigation'
    model_subdir = model_name.replace(' ', '/')
    model_directory = Path(models_directory, project_name, model_subdir)

    # change map we evaluate on
    map_name = 'AirSimNH'

    # read control params and make objects from config.py file
    config_path = Path(model_directory, 'config.py')

    # run config py file from local py script
    with open(config_path) as f:
        src = f.read()
        code = compile(src, "__main__.py", "exec")

        orig_argv = sys.argv[:]
        try:
            sys.argv = ['__main__.py', f'map_name:{map_name}', f'depth_sensor_name:DepthV1']
            exec(code)
        finally:
            sys.argv = orig_argv
        
    args = get_argparse()
        
    main(args.parse_args())
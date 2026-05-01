from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Environment
from map_tool_box.modules import Data_Map
# from map_tool_box.modules import Control
from map_tool_box.AirSimNNaviFI.Controller import Control
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
from pytorchfi.FI_Weights import FI_manager
from copy import deepcopy
from ultralytics import SAM


# ELLIPSE_CACHE = {}

# DEPTH_SHAPE = (144, 256)
# SCALING_FACTORS = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08,
#                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def precompute_ellipses(depth_shape, scaling_factors, thickness=2):
    h, w = depth_shape
    cx, cy = w // 2, h // 2

    y, x = np.ogrid[:h, :w]

    ellipse_map = {}

    for sf in scaling_factors:
        a = int(sf * w)
        b = int(sf * h)

        # Evita divisioni per zero
        if a <= thickness or b <= thickness:
            continue

        outer = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) <= 1
        inner = ((x - cx)**2 / (a - thickness)**2 +
                 (y - cy)**2 / (b - thickness)**2) <= 1

        ellipse_border = outer & (~inner)

        ellipse_map[sf] = (ellipse_border, inner)

    return ellipse_map


def get_argparse():
    parser = argparse.ArgumentParser(description='DQN configuration')
    parser.add_argument('--fsim_config', help='Fault simulation configuration json file path')
    parser.add_argument('--target_layer', help='The layer that the simulation has to target')
    parser.add_argument('--trials', help='Number of FI trials to run')
    parser.add_argument('--fsim_log_name', required=True, help='Directory name where the results are stored')
    parser.add_argument('--hardening', default=None, required=False, help='Name of the hardening Technique to implement')
    parser.add_argument('--paths_number', default=5, help='Number of paths to evaluate on (if not specified will evaluate on 5 paths per difficulty level)')
    parser.add_argument('--difficulties', default=None, help='Difficulty levels as a list (if not specified will evaluate on representative difficulty levels)')
    parser.add_argument('--sam', default=False, help='Analyze input complexity')
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

    if fsim_config:
        # sam_model = SAM("sam_b.pt") if bool(args.sam) else None
        with open(f'{fsim_config}', "r") as f:
            content = f.read()
            fsim_config=json.loads(content)
        print(fsim_config)
        
        full_log_path = deepcopy(args.fsim_log_name)
        
        num_episodes = len(spawner.difficulties) * n_paths

        FI_setup = FI_manager(log_path=full_log_path, chpt_file_name='ckpt_FI.json', fault_report_name='fsim_report.csv', num_episodes=num_episodes)
        FI_setup.open_golden_results("Golden_results")

        # print(configuration.controller._model._sb3model.policy.q_net)
        # print(args.hardening)
        
        if hardening:
            backup_dir=f"{Utils.get_global('repository_directory')}/AirSimNNaviFI/backup/{hardening}"
            Path(backup_dir).mkdir(parents=True, exist_ok=True)

        if hardening == 'Ranger':
            from map_tool_box.AirSimNNaviFI.Hardening.Ranger import implement_ranger
            layer_indices=[int(target_layer)]
            model = implement_ranger(model_UT=model, layers = layer_indices, output_dir=backup_dir, map_name=map_name, model_name=model_name)
        
        ELLIPSE_CACHE = precompute_ellipses(
            depth_shape=DEPTH_SHAPE,
            scaling_factors=SCALING_FACTORS,
            thickness=2
        )
        
        # output results here
        write_dir = 'Golden_results/'
        os.makedirs(write_dir, exist_ok=True)
        write_file = 'evaluation__test.p'
        write_path = Path(write_dir, write_file)
        accuracy, episodes = Control.eval(environment, model, write_path=write_path, save_observations=False, save_qvalues=True, run='Golden', FI_setup=FI_setup)
        
        FI_setup.close_golden_results()

        # sys.exit()

        # configuration.controller.run()
        
        # FAULT SIMULATION
        layer_types = [torch.nn.Conv2d, torch.nn.Linear]

        FI_setup.FI_framework.create_fault_injection_model(device=torch.device('cpu'),
                                                            model=model,
                                                            input_shape=[[3, 144, 256], [12,]],
                                                            layer_types=layer_types,
                                                            Neurons=True)

        FI_setup.generate_fault_list(flist_mode=fsim_config['fault_info']['neurons_rand_single_layer']['mode_inj'],
                                        f_list_file='fault_list.csv',
                                        trials= int(trials),
                                        layer=int(fsim_config['fault_info']['neurons_rand_single_layer']['layer']),
                                        bers = fsim_config['fault_info']['neurons_rand_single_layer']['bers'])
        
        

        
        for fault, k in FI_setup.iter_fault_list():
            write_dir = f'F_{k}_results/'
            write_file = 'evaluation__test.p'
            write_path = Path(write_dir, write_file)

            FI_setup.open_faulty_results(f"F_{k}_results")

            FI_setup.FI_framework.bit_flip_err_neuron(fault)
            # print(FI_setup.FI_framework.faulty_model)
            # sys.exit()
            accuracy, episodes = Control.eval(environment, model=FI_setup.FI_framework.faulty_model, write_path=write_path, save_observations=False, save_qvalues=True, fault=fault, FI_setup=FI_setup, run='Faulty')
            
            FI_setup.parse_results()

            # if k == 5:
            #     break

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
from map_tool_box.modules import Data_Structure
from map_tool_box.modules import Environment
from map_tool_box.modules import Data_Map
# from map_tool_box.modules import Control
from map_tool_box.NaviAPPFI.Controller import Control
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
from map_tool_box.NaviAPPFI.Hardening.FQ_ViT.models.ptq.layers import QConv2d, QLinear  

def get_argparse():
    parser = argparse.ArgumentParser(description='DQN configuration')
    parser.add_argument('--fsim_config', help='Fault simulation configuration json file path')
    parser.add_argument('--target_layer', help='The layer that the simulation has to target')
    parser.add_argument('--trials', help='The layer that the simulation has to target')
    parser.add_argument('--fsim_log_name', required=True, help='Directory name where the results are stored')
    parser.add_argument('--hardening', default='', help='Name of the hardening Technique to implement')
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
    n_paths = int(args.paths_number) # if None then will read all paths from file, otherwise an integer value specifying number of paths PER DIFFICULTY
    difficulties = args.difficulties # if None then will read all difficulties from file, otherwise expects a list of difficulty keys

    # read paths from file (usese some variables read in from config.py file) 
        # -- you can overwrite map_name or astar_version, but the default values or those used to train the model
    paths = Astar.read_curriculum(map_name, astar_version, set_name, n_paths, difficulties)

    # create spawner object to iterate through paths from environment
    spawner = Spawner.CurricululmEval(paths)
        
    # add any additional components to config
    others = []

    # create environment that we will step through (uses some objects read in from config.py file)
    environment = Environment.Episodic(data_map, spawner, actor, observer, terminators, others)

    if args.fsim_config:
        with open(f'{args.fsim_config}', "r") as f:
            content = f.read()
            fsim_config=json.loads(content)
        print(fsim_config)
        
        full_log_path = args.fsim_log_name
        
        num_episodes = len(spawner.difficulties) * n_paths

        FI_setup = FI_manager(log_path=full_log_path, chpt_file_name='ckpt_FI.json', fault_report_name='fsim_report.csv', num_episodes=num_episodes)
        FI_setup.open_golden_results("Golden_results")

        # print(configuration.controller._model._sb3model.policy.q_net)
        # print(args.hardening)

        if args.hardening == 'Ranger':
            backup_dir = f'reinforcement_learning/backup/{args.hardening}'
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Ranger import implement_ranger
            layer_indices=[args.hardened_layer]
            configuration = implement_ranger(configuration=configuration, layers = layer_indices, output_dir=backup_dir)
        elif args.hardening == 'MedianFilter':
            from NaviAPPFI.Hardening.MedianFilter import implement_median_filter
            configuration = implement_median_filter(configuration)
        elif args.hardening == 'FTClipAct':
            from NaviAPPFI.Hardening.FTClipAct import implement_FTClipAct
            configuration = implement_FTClipAct(configuration)
        elif args.hardening == 'Quantization':
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Quantization import apply_quantization
            from NaviAPPFI.Hardening.FQ_ViT.config import Config
            cfg = Config(ptf=False, lis=False, quant_method='minmax')
            configuration = apply_quantization(configuration, cfg=cfg)
        elif args.hardening == 'Best_model':
            backup_dir = f'reinforcement_learning/backup/{args.hardening}'
            os.makedirs(f'{backup_dir}', exist_ok=True)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Ranger import implement_ranger
            layers = [1,3,4,6]
            configuration = implement_ranger(configuration, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Quantization import apply_quantization
            from NaviAPPFI.Hardening.FQ_ViT.config import Config
            cfg = Config(ptf=False, lis=False, quant_method='minmax')
            layers = [0,2,5]
            configuration = apply_quantization(configuration, cfg=cfg, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.TMR import apply_tmr
            layers = [2,5,7]
            configuration = apply_tmr(configuration, layers=layers)
        elif args.hardening == 'Model1':
            backup_dir = f'reinforcement_learning/backup/{args.hardening}'
            os.makedirs(f'{backup_dir}', exist_ok=True)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Ranger import implement_ranger
            layers = [2,3]
            configuration = implement_ranger(configuration, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Quantization import apply_quantization
            from NaviAPPFI.Hardening.FQ_ViT.config import Config
            cfg = Config(ptf=False, lis=False, quant_method='minmax')
            layers = [1]
            configuration = apply_quantization(configuration, cfg=cfg, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.TMR import apply_tmr
            layers = [5,6,7]
            configuration = apply_tmr(configuration, layers=layers)
        elif args.hardening == 'Model2':
            backup_dir = f'reinforcement_learning/backup/{args.hardening}'
            os.makedirs(f'{backup_dir}', exist_ok=True)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Ranger import implement_ranger
            layers = [0,1,3]
            configuration = implement_ranger(configuration, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Quantization import apply_quantization
            from NaviAPPFI.Hardening.FQ_ViT.config import Config
            cfg = Config(ptf=False, lis=False, quant_method='minmax')
            layers = [4]
            configuration = apply_quantization(configuration, cfg=cfg, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.TMR import apply_tmr
            layers = [2,4,7]
            configuration = apply_tmr(configuration, layers=layers)
        elif args.hardening == 'Model3':
            backup_dir = f'reinforcement_learning/backup/{args.hardening}'
            os.makedirs(f'{backup_dir}', exist_ok=True)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Ranger import implement_ranger
            layers = [5]
            configuration = implement_ranger(configuration, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.Quantization import apply_quantization
            from NaviAPPFI.Hardening.FQ_ViT.config import Config
            cfg = Config(ptf=False, lis=False, quant_method='minmax')
            layers = [1,2,6]
            configuration = apply_quantization(configuration, cfg=cfg, layers=layers, output_dir=backup_dir)
            from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.TMR import apply_tmr
            layers = [1,2,4,6,7]
            configuration = apply_tmr(configuration, layers=layers)


        # print(configuration.controller._model._sb3model.policy.q_net)
        print(model.sb3model.q_net)

        # output results here
        write_dir = 'Golden_results/'
        os.makedirs(write_dir, exist_ok=True)
        write_file = 'evaluation__test.p'
        write_path = Path(write_dir, write_file)
        accuracy, episodes = Control.eval(environment, model, write_path=write_path, save_observations=False, run='Golden', FI_setup=FI_setup)

        FI_setup.close_golden_results()

        # sys.exit()

        # configuration.controller.run()
        
        # FAULT SIMULATION
        layer_types = [QConv2d, QLinear, torch.nn.Conv2d, torch.nn.Linear]

        FI_setup.FI_framework.create_fault_injection_model(device=torch.device('cpu'),
                                                            model=model,
                                                            input_shape=[[3, 144, 256], [12,]],
                                                            layer_types=layer_types,
                                                            Neurons=True)

        FI_setup.generate_fault_list(flist_mode=fsim_config['fault_info']['neurons_rand_single_layer']['mode_inj'],
                                        f_list_file='fault_list.csv',
                                        trials= int(fsim_config['fault_info']['neurons_rand_single_layer']['trials']),
                                        layer=int(fsim_config['fault_info']['neurons_rand_single_layer']['layer']),
                                        bers = fsim_config['fault_info']['neurons_rand_single_layer']['bers'])
        
        for fault, k in FI_setup.iter_fault_list():
            write_dir = f'F_{k}_results/'
            write_file = 'evaluation__test.p'
            write_path = Path(write_dir, write_file)

            FI_setup.open_faulty_results(f"F_{k}_results")

            FI_setup.FI_framework.bit_flip_err_neuron(fault)

            accuracy, episodes = Control.eval(environment, model=FI_setup.FI_framework.faulty_model, write_path=write_path, save_observations=False, fault=fault, FI_setup=FI_setup, run='Faulty')

            FI_setup.parse_results()

        configuration.disconnect_all()

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
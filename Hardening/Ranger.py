import torch
import os
from map_tool_box.AirSimNNaviFI.Hardening.ComputeTrainStats import *
import json
import sys

def apply_ranger_selective(UT_model, thrs={}, layers_to_replace=None, prefix=''):

    model = UT_model.sb3model.q_net
    idx = 0

    def _apply_thresholds(module, name_prefix=''):
        nonlocal idx
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if isinstance(child, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                if (layers_to_replace is None or idx in layers_to_replace) and full_name in thrs:
                    threshold = thrs[full_name]
                    new_layer = torch.nn.Sequential(
                        child,
                        torch.nn.Hardtanh(0, threshold)
                    )
                    setattr(module, name, new_layer)
                idx += 1
            else:
                _apply_thresholds(child, full_name)

    _apply_thresholds(model, prefix)
    return UT_model


def evaluate_thrs(model, map_name, model_name):

    thrs, _ , hook_handles = set_hooks(model)

    ## perform profiling through inference
    inference(model, map_name, model_name)

    remove_hooks(hook_handles)

    return thrs

def validate_thrs(configuration):

    if not os.path.exists(os.path.join(configuration.output_dir, 'ranger_thrs.json')):
        train_thrs = evaluate_thrs()
        with open(os.path.join(configuration.output_dir, 'ranger_thrs.json'), 'w') as f:
            f.write(json.dumps(train_thrs, indent=4))
    else:
        with open(os.path.join(configuration.output_dir, 'ranger_thrs.json'), 'r') as f:
            train_thrs = json.load(f)

    ## set hooks
    _, stats , hook_handles = set_hooks(configuration, train_thrs=train_thrs)

    ## perform profiling through inference
    inference(train_configuration = configuration)

    remove_hooks(hook_handles)

    return stats

def implement_ranger(model_UT, layers = None, output_dir = None, map_name = None, model_name = None):

    ## Estimate thresholds
    if not os.path.exists(os.path.join(output_dir, 'ranger_thrs.json')):
        train_thrs = evaluate_thrs(model_UT, map_name, model_name)
        with open(os.path.join(output_dir, 'ranger_thrs.json'), 'w') as f:
            f.write(json.dumps(train_thrs, indent=4))
    else:
        with open(os.path.join(output_dir, 'ranger_thrs.json'), 'r') as f:
            train_thrs = json.load(f)
    
    ## Set activation functions
    model = apply_ranger_selective(model_UT, thrs=train_thrs, layers_to_replace=layers)

    return model
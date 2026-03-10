import torch
import os
from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.ComputeTrainStats import *
import json

def apply_ranger_selective(UT_configuration, thrs={}, layers_to_replace=None, prefix=''):

    model = UT_configuration.controller._model._sb3model.policy.q_net
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
    return UT_configuration


def evaluate_thrs():

    train_configuration, output_dir = setup_inference_on_train()

    ## set hooks
    thrs, _ , hook_handles = set_hooks(train_configuration, output_dir=output_dir)

    ## perform profiling through inference
    inference(train_configuration)

    remove_hooks(hook_handles)

    train_configuration.disconnect_all()
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

def implement_ranger(configuration, layers = None, output_dir = None):

    ## Estimate thresholds
    if not os.path.exists(os.path.join(output_dir, 'ranger_thrs.json')):
        train_thrs = evaluate_thrs()
        with open(os.path.join(output_dir, 'ranger_thrs.json'), 'w') as f:
            f.write(json.dumps(train_thrs, indent=4))
    else:
        with open(os.path.join(output_dir, 'ranger_thrs.json'), 'r') as f:
            train_thrs = json.load(f)
    
    ## Set activation functions
    configuration = apply_ranger_selective(configuration, thrs=train_thrs, layers_to_replace=layers)

    return configuration
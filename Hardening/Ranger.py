import torch
import rl_utils as _utils
from configuration import Configuration
import math
import sys
import os
import numpy as np
import modules


def implement_ranger(configuration):

    ## Estimate thresholds
    thrs, fms = evaluate_thrs(configuration)
    print(thrs)
    ## Set activation functions
    configuration = apply_ranger(configuration, thrs=thrs, fms=fms)

    return configuration


def apply_ranger(configuration, thrs={}, prefix=''):
    model = configuration.controller._model._sb3model.policy.q_net
    def _apply_thresholds(module, name_prefix=''):
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(child, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)) and full_name in thrs:

                threshold = thrs[full_name]
                new_layer = torch.nn.Sequential(
                    child,
                    torch.nn.Hardtanh(0, threshold)
                )

                setattr(module, name, new_layer)

            else:
                _apply_thresholds(child, full_name)

    _apply_thresholds(model, prefix)

    return configuration


def evaluate_thrs(configuration):
    
    ## set hooks
    thrs, hook_handles = set_hooks(configuration)

    ## perform profiling through inference
    inference(configuration, hook_handles)
    return thrs

def set_hooks(configuration, prefix=''):
    hook_handles = []
    thrs = {}
    model = configuration.controller._model._sb3model.policy.q_net
    def _register(m, name_prefix=''):
        for name, layer in m.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                handle = layer.register_forward_hook(get_hook(full_name, thrs))
                hook_handles.append(handle)

            # Ricorsivamente continua con i figli
            _register(layer, full_name)

    _register(model, prefix)
    print(thrs)

    return thrs, hook_handles


def get_hook(name, thrs={}):
    def hook(module, input, output):
        # Get max from output tensor (regardless of shape)
        max_val = output.detach().max().item()
        if name not in thrs:
            thrs[name] = max_val
        else:
            thrs[name] = max(thrs[name], max_val)
    return hook

def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()

def inference(configuration, hook_handles):

    # RUN CONTROLLER
    configuration.controller.run()

    remove_hooks(hook_handles)

    # done
    _utils.speak('Evaluations done!')
    configuration.disconnect_all()
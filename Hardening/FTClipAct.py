import torch
import rl_utils as _utils
from configuration import Configuration
import math
import sys
import os
import numpy as np
import modules


def implement_FTClipAct(configuration):

    ## Estimate thresholds
    thrs = evaluate_thrs(configuration)
    print(thrs)
    ## Set activation functions
    configuration = apply_clip(configuration, thrs=thrs)

    return configuration

def apply_clip(configuration, thrs={}, prefix=''):
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

def get_hook(name, thrs={}):
    def hook(module, input, output):
        # Get max from output tensor (regardless of shape)
        max_val = output.detach().max().item()
        if name not in thrs:
            thrs[name] = max_val
        else:
            thrs[name] = max(thrs[name], max_val)
    return hook

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
    # print(thrs)

    return thrs, hook_handles


def evaluate_thrs(configuration):
    thrs, hook_handles = set_hooks(configuration)

    inference(configuration, hook_handles)

    tuned_thrs = {}

    for name, act_max in thrs.items():
        print(f"Tuning threshold for layer: {name}")
        best_T = threshold_fine_tuning(configuration, name, act_max)
        tuned_thrs[name] = best_T

    return tuned_thrs

def threshold_fine_tuning(configuration, layer_name, act_max, N=10, M=3, delta=0.01):


    def evaluate_auc(configuration, threshold):
        
        configuration = apply_clip(configuration, thrs={layer_name: threshold})
        successes = configuration.controller.run()['successes']
        goal_prob = sum(successes)/len(successes)
        return goal_prob

    counter = 1
    T_opt = 0
    S = [0, act_max]

    AUCs = []
    print(f'act_max: {act_max}')
    while counter <= N:
        T1 = S[0]
        T4 = S[1]
        T2 = T1 + (T4 - T1) / 3
        T3 = T2 + (T4 - T1) / 3

        thresholds = [T2, T3, T4]
        # print(thresholds)
        aucs = [evaluate_auc(configuration, t) for t in thresholds]

        AUCs.append(aucs)
        best_index = np.argmax(aucs)

        if best_index == 1:
            S = [T1, T2]
        elif best_index == 2:
            S = [T3, T4]
        else:
            S = [thresholds[best_index - 1], thresholds[best_index + 1]]

        T_opt = thresholds[best_index]

        # convergence check
        if counter >= M:
            diffs = [abs(aucs[i+1] - aucs[i]) for i in range(2)]
            if max(diffs) <= delta:
                break

        counter += 1

    return T_opt

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
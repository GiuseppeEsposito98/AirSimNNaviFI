import torch  
from copy import deepcopy
from NaviAPPFI.Hardening.FQ_ViT.models.ptq.layers import QConv2d, QLinear

class TMR(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.voters = torch.nn.ModuleList([deepcopy(layer) for _ in range(3)])
        for voter in self.voters:
            voter.load_state_dict(layer.state_dict()) 
    
    def forward(self, _in):
        props = [v(_in) for v in self.voters]
        outputs = torch.stack(props, dim=0)
        res = torch.mode(outputs, dim=0)[0]
        return res


def implement_tmr(model: torch.nn.Module, layers_to_replace=None):
    idx = 0  
    def _apply_tmr(module):
        nonlocal idx
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.Linear) or isinstance(child, QConv2d) or isinstance(child, QLinear):
                if layers_to_replace is None or idx in layers_to_replace:
                    # tmr_layers = torch.nn.Sequential(*[child for _ in range(3)])
                    # for i, layer in enumerate(tmr_layers):
                    #     layer.load_state_dict(child.state_dict())
                    tmr = TMR(child)
                    setattr(module, f"{name}", tmr)
                    # attach_hooks(tmr_layers)
                idx += 1
            else:
                _apply_tmr(child)

    _apply_tmr(model)

class TMROutputs:
    def __init__(self):
        self.outputs = None

    def add_output(self, output):
        if self.outputs is None:
            self.outputs = output.unsqueeze(0)
        else:
            print(self.outputs.shape)
            print(output.shape)
            self.outputs = torch.cat([self.outputs, output.unsqueeze(0)], dim=0)
    
    def clear_outputs(self):
        self.outputs = None


def attach_hooks(module):
    outputs = TMROutputs()
    
    def forward_hook(m, i, o):
        outputs.add_output(o)
    
    def majority_voting(m, i, o):
        res = torch.mode(outputs, dim=0)[0]
        outputs.clear_outputs()
        return res
    
    for _, child in module.named_children():
        child.register_forward_hook(forward_hook)
    
    module.register_forward_hook(majority_voting)


def apply_tmr(test_configuration, layers=None):

    # train_configuration.controller._model._sb3model.policy = 
    implement_tmr(model = test_configuration.controller._model._sb3model.policy.q_net, layers_to_replace=layers)
    
    return test_configuration
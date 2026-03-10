import json
import os
import torch
from NaviAPPFI.Hardening.FQ_ViT.models.ptq.layers import QConv2d, QLinear
from NaviAPPFI.Hardening.hard41293c68fe7e15560d26ba8fa6c1bf377a7df4fd.ComputeTrainStats import setup_inference_on_train    

def quantizable_nn(model: torch.nn.Module, cfg=None, layers_to_replace=None, quant_params=None):
    idx = 0  
    def _apply_quant_lyr(module):
        nonlocal idx
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Conv2d):
                device = next(child.parameters()).device
                if layers_to_replace is None or idx in layers_to_replace:
                    if idx in quant_params:
                        conv_dict = quant_params[idx]
                    else:
                        conv_dict = dict()
                        quant_params[idx] = conv_dict
                    new_layer = QConv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True if child.bias is not None else False,
                        bit_type=cfg.BIT_TYPE_A,
                        calibration_mode=cfg.CALIBRATION_MODE_A,
                        observer_str=cfg.OBSERVER_A,
                        quantizer_str=cfg.QUANTIZER_A,
                        quant_params = conv_dict
                    ).to(device)
                    new_layer.load_state_dict(child.state_dict())
                    setattr(module, name, new_layer)
                idx += 1

            elif isinstance(child, torch.nn.Linear):
                device = next(child.parameters()).device
                if layers_to_replace is None or idx in layers_to_replace:
                    if idx in quant_params:
                        linear_dict = quant_params[idx]
                    else:
                        linear_dict = dict()
                        quant_params[idx] = linear_dict
                    new_layer = QLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=True if child.bias is not None else False,
                        bit_type=cfg.BIT_TYPE_W,
                        calibration_mode=cfg.CALIBRATION_MODE_W,
                        observer_str=cfg.OBSERVER_W,
                        quantizer_str=cfg.QUANTIZER_W,
                        quant_params = linear_dict
                    ).to(device)
                    new_layer.load_state_dict(child.state_dict())
                    setattr(module, name, new_layer)
                idx += 1
            elif len(list(module.named_children())) > 0:
                _apply_quant_lyr(child)
            else:
                return quant_params

    _apply_quant_lyr(model)


def print_output_hook(module, input, output):
    if output != None:
        print(f"{module.__class__.__name__} output shape: {output.shape}")
    else:
        print(f"OUTPUT OF {module.__class__.__name__} IS NONE")

def attach_hooks_recursively(module):
    for name, child in module.named_children():
        # Se il modulo ha figli, esplora ricorsivamente
        if len(list(child.children())) > 0:
            attach_hooks_recursively(child)
        else:
            print(name)
            # Attacca l'hook al modulo foglia
            child.register_forward_hook(print_output_hook)

def obeserve_and_quantize(train_configuration, calib_iter = 10):
    # RUN CONTROLLER
    _, _, input_list = train_configuration.controller.run(calib_iter = calib_iter)

    Qmodel = train_configuration.controller._model
    Qmodel.model_open_calibrate()
    with torch.no_grad():
        for i, observation in enumerate(input_list):
            if i == len(input_list) - 1:
                Qmodel.model_open_last_calibrate()
            output = Qmodel.predict(observation)
        Qmodel.model_close_calibrate()
    Qmodel.model_quant()
    return Qmodel

def apply_quantization(test_configuration, calib_iter=10, layers=None, cfg=None, output_dir=None):
    if not os.path.exists(os.path.join(output_dir, "quant_params.json")):
        train_configuration, _ = setup_inference_on_train()
        quant_params = dict()
        # train_configuration.controller._model._sb3model.policy = 
        quantizable_nn(model = train_configuration.controller._model._sb3model.policy.q_net, cfg=cfg, layers_to_replace=layers, quant_params = quant_params)
        test_configuration.controller._model = obeserve_and_quantize(train_configuration, calib_iter=calib_iter)
        print(quant_params)
        with open(os.path.join(output_dir, "quant_params.json"), 'w') as f:
            f.write(json.dumps(quant_params, indent=4))
        train_configuration.disconnect_all()
    else:
        with open(os.path.join(output_dir, "quant_params.json"), 'r') as f:
            quant_params = json.load(f)
        print(quant_params)
        quantizable_nn(model = test_configuration.controller._model._sb3model.policy.q_net, cfg=cfg, layers_to_replace=layers, quant_params = quant_params)
        # test_configuration.controller._model = obeserve_and_quantize(test_configuration, calib_iter=calib_iter)
        print(test_configuration.controller._model._sb3model.policy.q_net)
    # else:

    return test_configuration
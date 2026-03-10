import torch
from torch import nn
from torch.nn.modules.utils import _single, _pair, _triple

import torch


class MedianPool(torch.nn.Module):

    def __init__(self, window_size) -> None:
        super(MedianPool, self).__init__()
        self.window_size = window_size  # int, non tensore

    def median_elimination_circular(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        dim = dim % x.ndim
        L = x.shape[dim]

        if self.window_size > 1:
            idx = [slice(None)] * x.ndim
            idx[dim] = slice(0, self.window_size-1)
            circular_pad = x[tuple(idx)]
            x_padded = torch.cat([x, circular_pad], dim=dim)
        else:
            x_padded = x

        if dim != x.ndim - 1:
            x_padded = x_padded.movedim(dim, -1)

        windows = x_padded.unfold(-1, self.window_size, 1)
        vals = windows.clone()
        n = self.window_size

        median_vals = vals.median(dim = -1).values

        if dim != x.ndim - 1:
            median_vals = median_vals.movedim(-1, dim)

        return median_vals

    def forward(self, input_):
        return self.median_elimination_circular(x=input_)

    

def myconv2d(input_, weight, pooling_op, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=0):
    batch_size, in_channels, in_h, in_w = input_.shape
    out_channels, _, kh, kw = weight.shape

    out_h = (in_h - kh + 2 * padding[0]) // stride[0] + 1
    out_w = (in_w - kw + 2 * padding[1]) // stride[1] + 1

    input_pool = pooling_op(input_) 

    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inps_unf = unfold(input_pool)
    
    w_ = weight.view(weight.size(0), -1).transpose(0,1).to(input_.device)

    if bias is not None:
        bias = bias.to(input_.device)
        out_unf = inps_unf.transpose(1,2).matmul(w_) + bias
    else:
        out_unf = inps_unf.transpose(1,2).matmul(w_)

    out_unf = out_unf.transpose(1,2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)

    return out

class MyConv2D(torch.nn.modules.conv._ConvNd):
    """
    Implements a standard convolution layer that can be used as a regular module
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, padding_mode='zeros', groups=1, output_padding=None, bias_tensor=None):

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        kernel_size = _pair(kernel_size)
        super(MyConv2D, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            transposed=False, output_padding=output_padding, groups=groups, bias=self.bias, padding_mode=padding_mode)
        self.pooling_op = MedianPool(window_size=3)
        

    def conv2d_forward(self, input_):
        return myconv2d(input=input_, weight=self.weight, pooling_op=self.pooling_op, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

    def forward(self, input_):
        return self.conv2d_forward(input_)

def implement_median_filter(configuration):

    model = configuration.controller._model._sb3model.policy.q_net

    def apply_median_filter(model):
        for name, module in model.named_children():
            print(type(name))
            
            if name == "0" and not isinstance(module, nn.Linear):
                pass
            if isinstance(module, nn.Conv2d):
                # print(module.bias.shape)
                conv2d = MyConv2D(in_channels = module.in_channels, out_channels= module.out_channels, kernel_size=module.kernel_size, \
                                stride=module.stride,padding=module.padding, dilation=module.dilation,bias=True if module.bias != None else False, \
                                padding_mode=module.padding_mode, groups=module.groups, weight=module.weight, \
                                output_padding=module.output_padding, bias_tensor=module.bias)
                setattr(model, name, conv2d)
            elif isinstance(module, nn.Linear):
                median_pool = MedianPool(window_size=3)
                new_layer = nn.Sequential(*[median_pool, module])
                setattr(model, name, new_layer)
            else: 
                apply_median_filter(module)
    
    apply_median_filter(model)

    return configuration
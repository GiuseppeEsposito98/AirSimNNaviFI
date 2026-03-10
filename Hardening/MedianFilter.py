import torch
from torch import nn
from torch.nn.modules.utils import _single, _pair, _triple

class MedianPool2D(torch.nn.Module):

    def __init__(self, stride, window_size) -> None:
        super(MedianPool2D, self).__init__()
        self.stride=torch.tensor(stride)
        self.window_size = torch.tensor(window_size)

    def forward(self, input):
        ic = torch.tensor(input.size()[1])
        ph = torch.add(torch.mul(torch.sub(ic,1),self.stride),self.window_size)
        # padding depth
        padding = torch.sub(ph,ic)

        if padding != 0:
            padding = input[:,:padding,:,:]
            x = torch.cat((input, padding), dim=1)
        else: x = input

        x = x.unfold(dimension=1, size=3, step=1)
        output, indices = x.median(dim=-1)
        return output
    



class MedianPool1d(torch.nn.Module):
    def __init__(self, stride, window_size) -> None:
        super(MedianPool1d, self).__init__()
        self.stride=stride
        self.window_size = window_size
    
    def forward(self, input):
        ic = input.size()[1]
        ph = ((ic-1)*self.stride)+self.window_size

        # padding depth
        padding = ph - ic

        if padding != 0:
            padding = input[:,:padding]
            x = torch.cat((input, padding), dim=1)
        else: x = input
        x = x.unfold(dimension=1, size=3, step=1)
        output, indices = x.median(dim=-1)
        # x = x.contiguous().view(x.size()[:2] + (-1,))
        
        return output
    
def myconv2d(input, weight, pooling_op, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=0):
    """
    Function to process an input with a standard convolution
    """
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kh, kw = weight.shape
    in_h = torch.tensor(in_h)
    kh = torch.tensor(kh)
    out_h = int(torch.add(torch.div(torch.add(torch.sub(in_h, kh), torch.mul(torch.tensor(2), torch.tensor(padding[0]))), torch.tensor(stride[0])), torch.tensor(1)))
    out_w = int(torch.add(torch.div(torch.add(torch.sub(in_w, kw), torch.mul(torch.tensor(2), torch.tensor(padding[1]))), torch.tensor(stride[1])), torch.tensor(1)))
    # print(out_h)
    # print(out_w)
    input_pool = pooling_op(input)

    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inps_unf=unfold(input_pool)
    # print(inps_unf.shape)
    w_ = weight.view(weight.size(0), -1).transpose(0,1)
    # print(w_.shape)
    if bias is not None:
        out_unf = inps_unf.transpose(1,2).matmul(w_).transpose(1, 2)
    else:
        out_unf = torch.add(inps_unf.transpose(1,2).matmul(w_), bias).transpose(1, 2)
    
    out = out_unf.view(batch_size, out_channels, out_h, out_w)

    return out.float()

class MyConv2D(torch.nn.modules.conv._ConvNd):
    """
    Implements a standard convolution layer that can be used as a regular module
    """
    def __init__(self, in_channels, out_channels, kernel_size, weight, stride=1,
                 padding=0, dilation=1,
                 bias=False, padding_mode='zeros', groups=0, output_padding=None, bias_tensor=None):

        super(MyConv2D, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            transposed=False, output_padding=output_padding, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bias_tensor = bias_tensor
        self.weight = weight
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        kernel_size = _pair(kernel_size)
        self.pooling_op = MedianPool2D(stride=1, window_size=3)
        

    def conv2d_forward(self, input, weight):
        return myconv2d(input=input, weight=weight, pooling_op=self.pooling_op, bias=self.bias_tensor, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

def implement_median_filter(configuration):

    model = configuration.controller._model._sb3model.policy.q_net

    def apply_median_filter(model):
        for name, module in model.named_children():
            print(type(name))
            
            if name == "0" and not isinstance(module, nn.Linear):
                pass
            elif isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    conv2d = MyConv2D(in_channels = module.in_channels, out_channels= module.out_channels, kernel_size=module.kernel_size, \
                                    stride=module.stride,padding=module.padding, dilation=module.dilation,bias=True, \
                                    padding_mode=module.padding_mode, groups=module.groups, weight=module.weight, \
                                    output_padding=module.output_padding, bias_tensor=module.bias)
                else: 
                    conv2d = MyConv2D(in_channels = module.in_channels, out_channels= module.out_channels, kernel_size=module.kernel_size, \
                                    stride=module.stride,padding=module.padding, dilation=module.dilation,bias=False, \
                                    padding_mode=module.padding_mode, groups=module.groups, weight=module.weight, \
                                    output_padding=module.output_padding)
                setattr(model, name, conv2d)
            elif isinstance(module, nn.Linear):
                median_pool = MedianPool1d(window_size=3, stride=1)
                new_layer = nn.Sequential(*[median_pool, module])
                setattr(model, name, new_layer)
            else: 
                apply_median_filter(module)
    
    apply_median_filter(model)

    return configuration
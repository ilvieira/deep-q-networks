import torch.nn.functional as F
import torch


def clip_mse(net_input, target, clip_value=1):
    # assumes the value and target are 2-dimensional tensors
    def clip(t):
        return clip_value if t > clip_value else -clip_value if t < -clip_value else t

    ten = (net_input - target)**2
    n = ten.shape[0]
    for i in range(n):
        ten[i] = clip(ten[i])
    return ten.mean()


def clip_mse2(net_input, target, clip_value=1):
    # assumes the value and target are 2-dimensional tensors
    def clip(t):
        return clip_value if t > clip_value else -clip_value if t < -clip_value else t

    ten = net_input-target
    n = ten.shape[0]

    for i in range(n):
        ten[i] = clip(ten[i])

    new_target = target-target

    return F.mse_loss(ten, new_target)


def clip_mse3(net_input, target, clip_value=1, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device is None else torch.device(device)

    shape = net_input.shape
    comp1 = torch.ones(shape).to(device)*clip_value
    compm1 = -torch.ones(shape).to(device)*clip_value
    new_target = torch.zeros(shape).to(device)
    new_input = torch.maximum(compm1, torch.minimum(comp1, net_input-target))

    return F.mse_loss(new_input, new_target)

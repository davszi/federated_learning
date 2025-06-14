from collections import OrderedDict

import torch
import torch.nn as nn
from flwr.common import NDArrays
from torchvision.models import vgg11, resnet18


def Net(architecture: str) -> nn.Module:
    architecture = architecture.lower()
    match architecture:
        case "vgg11":
            return vgg11(weights=None)
        case "resnet18":
            return resnet18(weights=None)
        case _:
            raise ValueError(f"Unsupported network architecture: {architecture}")

def get_weights(net) -> NDArrays:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
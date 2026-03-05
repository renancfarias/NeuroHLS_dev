import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from primitive_debug import debug_net, run_cpp_impl

net = torch.nn.Sequential(
    torch.nn.Linear(5, 2)
)

with torch.no_grad():
    net[0].weight = torch.nn.Parameter(torch.tensor([
        [2.3, 3.2, -1.7, 2.0, 4.5],
        [-0.9, 1.3, 2.6, 5.4, 0.4]
    ]))
    
    net[0].bias = torch.nn.Parameter(torch.tensor([0.5, -0.5]))

debug_net(net, [torch.tensor([[1., 2., 3., 4., 5.]]), torch.tensor([[3., 5., 7., 9., 11.]])], "affine")
run_cpp_impl()
import sys
from pathlib import Path
import numpy as np

# adiciona a pasta pai ao sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from primitive_debug import debug_net, run_cpp_impl, save_out_py

def debug_conv2d():

    net = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=(2, 3),
            stride=1,
            padding=0,
            dilation=(2, 1),
            groups=2,
            bias=True
        )
    )

    with torch.no_grad():
        net[0].weight = torch.nn.Parameter(torch.tensor([
            # ----- Grupo 1 -----
            [  # out_channel 0
                [[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]],

                [[-0.1, -0.2, -0.3],
                [-0.4, -0.5, -0.6]],
            ],
            [  # out_channel 1
                [[0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2]],

                [[-0.7, -0.8, -0.9],
                [-1.0, -1.1, -1.2]],
            ],

            # ----- Grupo 2 -----
            [  # out_channel 2
                [[0.05, 0.10, 0.15],
                [0.20, 0.25, 0.30]],

                [[-0.05, -0.10, -0.15],
                [-0.20, -0.25, -0.30]],
            ],
            [  # out_channel 3
                [[0.33, 0.44, 0.55],
                [0.66, 0.77, 0.88]],

                [[-0.33, -0.44, -0.55],
                [-0.66, -0.77, -0.88]],
            ],
        ], dtype=torch.float32))

    net[0].bias = torch.nn.Parameter(
        torch.tensor([0.5, -0.5, 0.25, -0.25], dtype=torch.float32)
    )

    input = torch.tensor(
    [
        [
            # Canal 0
            [
                [1., 2., 3., 4., 5.],
                [5., 4., 3., 2., 1.],
                [1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0.],
                [2., 2., 2., 2., 2.],
            ],

            # Canal 1
            [
                [0., 1., 0., 1., 0.],
                [1., 0., 1., 0., 1.],
                [0., 1., 0., 1., 0.],
                [1., 0., 1., 0., 1.],
                [0., 1., 0., 1., 0.],
            ],

            # Canal 2
            [
                [2., 2., 2., 2., 2.],
                [3., 3., 3., 3., 3.],
                [4., 4., 4., 4., 4.],
                [5., 5., 5., 5., 5.],
                [6., 6., 6., 6., 6.],
            ],

            # Canal 3
            [
                [9., 8., 7., 6., 5.],
                [4., 3., 2., 1., 0.],
                [1., 2., 3., 4., 5.],
                [6., 7., 8., 9., 0.],
                [1., 3., 5., 7., 9.],
            ],
        ]
    ], dtype=torch.float32)

    net.eval()

    txt_str = f"Sample 1:\n\n{np.round(net(input)[0].detach().numpy(), 4)}\n"
    save_out_py(txt_str)

debug_conv2d()

def debug_conv2d_2():

    net = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2,2),
            stride=1,
            padding="same")
    )

    # Definindo pesos manualmente
    with torch.no_grad():
        net[0].weight = torch.nn.Parameter(
            torch.tensor([[[[1.0, 2.0],
                            [3.0, 4.0]]]])
        )

        net[0].bias = torch.nn.Parameter(
            torch.tensor([1.0])
        )

    # Tensor de entrada (batch=1, channel=1, height=3, width=3)
    x = torch.tensor(
        [[[[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]]]]
    )

    output = net(x)
    print(output)

run_cpp_impl()
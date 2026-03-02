import snntorch as snn
import torch
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir

def debug_net(net, sample_data, primitive_name):

    if not isinstance(sample_data, (list, tuple)):
        nir_sample_data = sample_data
        sample_data = [sample_data]
    else:
        nir_sample_data = sample_data[0]

    nir_graph = export_to_nir(net, nir_sample_data, model_name=primitive_name)
    
    net_from_nir = import_from_nir(nir_graph)
    net_from_nir.eval()

    print(f"\n*** {primitive_name} ***")

    print("\n" + "-" * 30 + "\nNIR Graph\n" + "-" * 30 + "\n")
    print(net_from_nir)

    # ----------------------------------------------------------

    print("\n" + "-" * 30 + "\nLayer Weights\n" + "-" * 30 + "\n")
    
    for name, layer in net_from_nir.named_children():

        if name == "input" or name == "output":
            continue

        print(f" - {layer}\n")

        for name, param in layer.named_parameters():
            print(f"\t{name}:")

            for tensor in param:
                print(f"\t\t{tensor.detach()}")

    # ----------------------------------------------------------

    print("\n" + "-" * 30 + "\nSamples\n" + "-" * 30 + "\n")

    print(f"Shape: {sample_data[0].shape[1:]}\n")

    for data in sample_data:
        print(data)

    # ----------------------------------------------------------

    print("\n" + "-" * 30 + "\nOutputs\n" + "-" * 30 + "\n")

    print(f"Shape: {net_from_nir(sample_data[0])[0].shape[1:]}\n")

    for data in sample_data:
        output = net_from_nir(data)
        print(output[0].detach())
    
    print()

def debug_linear():

    net = torch.nn.Sequential(
        torch.nn.Linear(5, 2)
    )

    with torch.no_grad():
        net[0].weight = torch.nn.Parameter(torch.tensor([
            [2.3, 3.2, -1.7, 2.0, 4.5],
            [-0.9, 1.3, 2.6, 5.4, 0.4]
        ]))
        
        net[0].bias = torch.nn.Parameter(torch.tensor([0.5, -0.5]))

    debug_net(net, [torch.tensor([[1., 2., 3., 4., 5.]]), torch.tensor([[3., 5., 7., 9., 11.]])], "linear")

debug_linear()


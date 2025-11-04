"""
Fixed snnTorch to NIR export script

This script exports a snnTorch sequential network to NIR format.
The original export_to_nir function had compatibility issues with NIR 1.0.6,
specifically missing the required 'v_reset' parameter for LIF neurons.

This version manually creates the NIR graph with proper parameter handling.

Network architecture:
- Input: 784 (flattened 28x28)
- Linear: 784 → 500
- LIF neuron (beta=0.9)
- Linear: 500 → 10  
- LIF neuron (beta=0.9)
- Output: 10 classes
"""

import snntorch as snn
import torch
import nir
import numpy as np

def create_nir_lif_from_snntorch(snn_lif, input_size):
    """Convert snnTorch Leaky neuron to NIR LIF with proper parameters."""
    # Extract parameters from snnTorch neuron
    beta = float(snn_lif.beta)
    threshold = float(snn_lif.threshold)
    
    # Convert beta to tau (time constant)
    # In snnTorch: mem = beta * mem + input
    # In NIR: tau is the membrane time constant
    tau = np.array([1.0 / (1.0 - beta)] * input_size)
    
    # Set resistance (typically 1.0 for simple cases)
    r = np.array([1.0] * input_size)
    
    # Set leak potential (typically 0.0)
    v_leak = np.array([0.0] * input_size)
    
    # Set threshold voltage
    v_threshold = np.array([threshold] * input_size)
    
    # Set reset voltage based on reset mechanism
    if snn_lif.reset_mechanism == "subtract":
        v_reset = np.array([0.0] * input_size)  # Reset to 0 after subtracting threshold
    elif snn_lif.reset_mechanism == "zero":
        v_reset = np.array([0.0] * input_size)  # Hard reset to 0
    else:  # "none"
        v_reset = np.array([0.0] * input_size)  # Default to 0
    
    return nir.LIF(
        tau=tau,
        r=r,
        v_leak=v_leak,
        v_threshold=v_threshold,
        v_reset=v_reset
    )

# Create snnTorch network
lif1 = snn.Leaky(beta=0.9, init_hidden=True)
lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

snntorch_network = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 500),
    lif1,
    torch.nn.Linear(500, 10),
    lif2
)

sample_data = torch.randn(1, 784)

# Create NIR graph manually
nodes = {}

# Add input node
nodes["input"] = nir.Input(input_type={"input": np.array([784])})

# Add flatten layer (implicit in the linear layer input size)
# Add first linear layer
linear1_weight = snntorch_network[1].weight.detach().numpy()
linear1_bias = snntorch_network[1].bias.detach().numpy() if snntorch_network[1].bias is not None else None
nodes["linear1"] = nir.Affine(weight=linear1_weight, bias=linear1_bias)

# Add first LIF neuron
nodes["lif1"] = create_nir_lif_from_snntorch(lif1, 500)

# Add second linear layer
linear2_weight = snntorch_network[3].weight.detach().numpy()
linear2_bias = snntorch_network[3].bias.detach().numpy() if snntorch_network[3].bias is not None else None
nodes["linear2"] = nir.Affine(weight=linear2_weight, bias=linear2_bias)

# Add second LIF neuron
nodes["lif2"] = create_nir_lif_from_snntorch(lif2, 10)

# Add output node
nodes["output"] = nir.Output(output_type={"output": np.array([10])})

# Define edges (connections between nodes)
edges = [
    ("input", "linear1"),
    ("linear1", "lif1"),
    ("lif1", "linear2"),
    ("linear2", "lif2"),
    ("lif2", "output")
]

# Create NIR graph
nir_model = nir.NIRGraph(nodes=nodes, edges=edges)

print("NIR model created successfully!")
print(f"Nodes: {list(nir_model.nodes.keys())}")
print(f"Edges: {nir_model.edges}")

# Salva
nir.write("nir_model.nir", nir_model)
print("NIR model saved as 'nir_model.nir'")

import snntorch as snn
import torch
from snntorch.export_nir import export_to_nir
import nir

fc1 = torch.nn.Linear(784, 500)
lif1 = snn.Leaky(
    beta=torch.full((fc1.out_features,), 0.9),
    threshold=torch.ones(fc1.out_features),
    init_hidden=True
)
fc2 = torch.nn.Linear(500, 10)
lif2 = snn.Leaky(
    beta=torch.full((fc2.out_features,), 0.9),
    threshold=torch.ones(fc2.out_features),
    init_hidden=True,
    output=True
)

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    fc1, lif1,
    fc2, lif2
)

sample_data = torch.randn(1, 784)
nir_graph = export_to_nir(net, sample_data, ignore_dims=[0])  # 👈 chave
nir.write("teste.nir", nir_graph)
#print(nir_graph)    

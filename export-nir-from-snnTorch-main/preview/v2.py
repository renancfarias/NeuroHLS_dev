import torch, torch.nn as nn
import snntorch as snn
from snntorch.export_nir import export_to_nir
import nir

# Modelo idêntico ao seu
lif1 = snn.Leaky(beta=0.9, init_hidden=True)
lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 500),
    lif1,
    nn.Linear(500, 10),
    lif2
)

# ✅ Dê um batch >1 ou =1 mas 2D/4D. Exs:

# Opção A: imagens 28x28 (o Flatten cuida do resto)
sample_data = torch.randn(8, 1, 28, 28)   # batch=8

# Opção B: já achatado
# sample_data = torch.randn(8, 784)

nir_model = export_to_nir(model, sample_data)
nir.write("nir_model.nir", nir_model)

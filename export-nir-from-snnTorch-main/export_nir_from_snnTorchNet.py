# tutorial6_snntorch_nir.py
# Tutorial 6 + Exportação para NIR (robusta)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

# --- NIR / NIRTorch ---
import nir
from snntorch.export_nir import export_to_nir
import numpy as np
import math

# =========================
# Dispositivo e parâmetros
# =========================
batch_size = 128
data_path = "./data/mnist"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

# =========================
# DataLoaders (MNIST)
# =========================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

mnist_train = datasets.MNIST(data_path, train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(mnist_test,  batch_size=batch_size, shuffle=True, drop_last=True)

#===========
# LIF:
#===========
lif1 = snn.Leaky(
    beta=torch.full((12, 12, 12), 0.5, device=device),
    threshold=torch.ones(12, 12, 12, device=device),
    spike_grad=spike_grad, init_hidden=True
)

# depois de conv(12,64,5) + avgpool(2) → (64,4,4)
lif2 = snn.Leaky(
    beta=torch.full((64, 4, 4), 0.5, device=device),
    threshold=torch.ones(64, 4, 4, device=device),
    spike_grad=spike_grad, init_hidden=True
)

# último Leaky (após Linear 1024→10) pode ser vetor (10,)
lif3 = snn.Leaky(
    beta=torch.full((10,), 0.5, device=device),
    threshold=torch.ones(10, device=device),
    spike_grad=spike_grad, init_hidden=True, output=True
)

# =========================
# Arquitetura CSNN:
# 12C5-MP2-64C5-MP2-1024FC10
# =========================
net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.AvgPool2d(2),
    lif1,
    nn.Conv2d(12, 64, 5),
    nn.AvgPool2d(2),
    lif2,
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
    lif3
).to(device)

# =========================
# Forward temporal
# =========================
def forward_pass(net, num_steps, data):
    mem_rec, spk_rec = [], []
    utils.reset(net)
    for _ in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

# =========================
# Loss & métrica
# =========================
loss_fn = SF.ce_rate_loss()

@torch.no_grad()
def batch_accuracy(loader, net, num_steps):
    total = 0
    acc_sum = 0.0
    net.eval()
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_rec, _ = forward_pass(net, num_steps, data)
        acc_sum += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total  += spk_rec.size(1)
    return acc_sum / total

# =========================
# Treinamento
# =========================
def train(num_epochs=1, lr=1e-2, log_every=50):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    counter = 0
    for epoch in range(num_epochs):
        net.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)
            loss_val = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if counter % log_every == 0:
                net.eval()
                with torch.no_grad():
                    test_acc = batch_accuracy(test_loader, net, num_steps)
                print(f"Iter {counter:5d} | Test Acc: {test_acc*100:6.2f}% | Loss: {loss_val.item():.3f}")
                net.train()
            counter += 1

# =========================
# Execução
# =========================
if __name__ == "__main__":
    print(f"Device: {device}")

    # Sanity check
    data0, targets0 = next(iter(train_loader))
    data0, targets0 = data0.to(device), targets0.to(device)
    spk0, _ = forward_pass(net, num_steps, data0)
    loss0 = loss_fn(spk0, targets0)
    print(f"Loss inicial (não treinado) ~ {loss0.item():.3f}")

    # 1) Treinar (ajuste épocas conforme GPU/tempo)
    train(num_epochs=1, lr=1e-2, log_every=50)

    # 2) Avaliar acurácia final
    test_acc = batch_accuracy(test_loader, net, num_steps)
    print(f"Acurácia final (após treinamento): {test_acc*100:6.2f}%")

    # 3) Exportar para NIR e salvar em disco (faz no CPU)
    import copy

    net.eval()
    utils.reset(net)

    # Copia segura no CPU para evitar .numpy() em tensores CUDA/MPS
    net_cpu = copy.deepcopy(net).to("cpu")
    utils.reset(net_cpu)

    # Gere uma amostra no CPU (B, C, H, W). Pode ser randômica ou vinda do loader.
    # Opção A: randômica
    sample_cpu = torch.randn(1, 1, 28, 28)

    # Opção B: usar um batch real
    # data_real, _ = next(iter(train_loader))
    # sample_cpu = data_real[:1].cpu()

    nir_graph = export_to_nir(net_cpu, sample_cpu, ignore_dims=[0])
    nir.write("csnn_mnist.nir", nir_graph)
    print("NIR salvo em: csnn_mnist.nir")



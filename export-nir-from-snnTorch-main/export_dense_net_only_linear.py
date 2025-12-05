# export_dense_net_only_linear.py
# Rede totalmente conectada (apenas Linear, sem ativações) + Exportação para NIR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- NIR ---
import nir
from snntorch.export_nir import export_to_nir
import copy

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

# =========================
# Arquitetura Dense (apenas Linear):
# 784 → 256 → 128 → 10
# =========================
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.Linear(256, 128),
    nn.Linear(128, 10)
).to(device)

# =========================
# Loss & métrica
# =========================
loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def batch_accuracy(loader, net):
    total = 0
    correct = 0
    net.eval()
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return correct / total

# =========================
# Treinamento
# =========================
def train(num_epochs=1, lr=1e-3, log_every=50):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    counter = 0
    for epoch in range(num_epochs):
        net.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            output = net(data)
            loss_val = loss_fn(output, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if counter % log_every == 0:
                test_acc = batch_accuracy(test_loader, net)
                print(f"Iter {counter:5d} | Test Acc: {test_acc*100:6.2f}% | Loss: {loss_val.item():.3f}")
            counter += 1

# =========================
# Execução
# =========================
if __name__ == "__main__":
    print(f"Device: {device}")
    print("Arquitetura: 784 → 256 → 128 → 10 (apenas Linear, sem ativações)")

    # Sanity check
    data0, targets0 = next(iter(train_loader))
    data0, targets0 = data0.to(device), targets0.to(device)
    output0 = net(data0)
    loss0 = loss_fn(output0, targets0)
    print(f"Loss inicial (não treinado) ~ {loss0.item():.3f}")

    # 1) Treinar
    train(num_epochs=1, lr=1e-3, log_every=50)

    # 2) Avaliar acurácia final
    test_acc = batch_accuracy(test_loader, net)
    print(f"Acurácia final (após treinamento): {test_acc*100:6.2f}%")

    # 3) Exportar para NIR
    net.eval()

    # Copia no CPU para exportar
    net_cpu = copy.deepcopy(net).to("cpu")

    # Amostra de entrada (batch_size=1, canais=1, altura=28, largura=28)
    sample_cpu = torch.randn(1, 1, 28, 28)

    try:
        nir_graph = export_to_nir(net_cpu, sample_cpu, ignore_dims=[0])
        nir.write("dense_only_linear.nir", nir_graph)
        print("✓ NIR salvo em: dense_only_linear.nir")
        print("\nNós do grafo NIR:")
        for node_name, node in nir_graph.nodes.items():
            print(f"  - {node_name}: {type(node).__name__}")
    except Exception as e:
        print(f"✗ Exportação NIR falhou: {e}")
        print("\nMotivo: O export_to_nir do snnTorch precisa de neurônios spiking (LIF, Synaptic, etc.)")
        print("As camadas Linear/Conv2d são exportadas apenas como conexões entre neurônios spiking.")
        
        # Salvar como PyTorch
        torch.save(net_cpu.state_dict(), "dense_only_linear.pt")
        print("\nModelo PyTorch salvo em: dense_only_linear.pt")

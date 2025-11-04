# Spiking Neural Network NIR Export - snnTorch Implementation

This project demonstrates how to train a Convolutional Spiking Neural Network (CSNN) using [snnTorch](https://snntorch.readthedocs.io/) and export it to the [Neuromorphic Intermediate Representation (NIR)](https://neuroir.org/) format for deployment on neuromorphic hardware.

## 🎯 Project Overview

Based on [snnTorch Tutorial 6](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html), this implementation extends the original tutorial by adding robust NIR export capabilities, enabling trained spiking neural networks to be deployed on neuromorphic computing platforms like Intel's Loihi, SpiNNaker, and others.

## 🧠 Network Architecture

The implemented CSNN follows the architecture: **12C5-MP2-64C5-MP2-1024FC10**

```
Input (28×28) → Conv2d(1→12, kernel=5) → AvgPool2d(2) → LIF → 
Conv2d(12→64, kernel=5) → AvgPool2d(2) → LIF → 
Flatten → Linear(1024→10) → LIF → Output
```

- **Dataset**: MNIST (28×28 grayscale images)
- **Neuron Model**: Leaky Integrate-and-Fire (LIF)
- **Time Steps**: 50
- **Learning**: Surrogate gradient descent with fast sigmoid

## 📋 Requirements

```bash
pip install torch==2.8.0 torchvision
pip install snntorch==0.9.4
pip install nir==1.0.6 nirtorch==2.0.5
```

### Tested Versions
- **PyTorch**: 2.8.0
- **snnTorch**: 0.9.4  
- **NIRTorch**: 2.0.5
- **NIR**: 1.0.6

## 🚀 Key Features

### 1. **Temporal Forward Pass**
```python
def forward_pass(net, num_steps, data):
    mem_rec, spk_rec = [], []
    utils.reset(net)
    for _ in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)
```

### 2. **Rate-Based Loss Function**
Uses `SF.ce_rate_loss()` for spike-rate based cross-entropy loss, suitable for temporal spike trains.

### 3. **NIR Export Capability**
Exports trained models to NIR format for neuromorphic hardware deployment.

## 🔧 NIR Export Modifications

The main modifications required to enable NIR export are implemented in `export_nir_from_snnTorchNet.py`:

### **1. Explicit LIF Neuron Definitions**

**❌ Original (Tutorial 6):**
```python
net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.AvgPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    # ...
)
```

**✅ Modified for NIR:**
```python
# Define LIF neurons with explicit dimensions
lif1 = snn.Leaky(
    beta=torch.full((12, 12, 12), 0.5, device=device),
    threshold=torch.ones(12, 12, 12, device=device),
    spike_grad=spike_grad, init_hidden=True
)

lif2 = snn.Leaky(
    beta=torch.full((64, 4, 4), 0.5, device=device),
    threshold=torch.ones(64, 4, 4, device=device),
    spike_grad=spike_grad, init_hidden=True
)

lif3 = snn.Leaky(
    beta=torch.full((10,), 0.5, device=device),
    threshold=torch.ones(10, device=device),
    spike_grad=spike_grad, init_hidden=True, output=True
)

# Use the defined instances
net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.AvgPool2d(2),
    lif1,  # Use instance instead of inline definition
    nn.Conv2d(12, 64, 5),
    nn.AvgPool2d(2),
    lif2,
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
    lif3
)
```

### **2. CPU Copy for NIR Export**

```python
# Export to NIR (must be done on CPU)
import copy

net.eval()
utils.reset(net)

# Safe CPU copy to avoid .numpy() issues with CUDA/MPS tensors
net_cpu = copy.deepcopy(net).to("cpu")
utils.reset(net_cpu)

# Generate CPU sample for tracing
sample_cpu = torch.randn(1, 1, 28, 28)

# Export to NIR
nir_graph = export_to_nir(net_cpu, sample_cpu, ignore_dims=[0])
nir.write("csnn_mnist.nir", nir_graph)
```

### **3. Key Technical Details**

#### **Why Explicit Dimensions?**
- NIR requires explicit tensor shapes for neuron states
- `beta` and `threshold` must be tensors matching the output dimensions
- Enables proper state mapping in neuromorphic hardware

#### **Why CPU Copy?**
- `export_to_nir()` internally uses `.numpy()` conversion
- CUDA/MPS tensors cannot be directly converted to NumPy
- `copy.deepcopy(net).to("cpu")` ensures all parameters are on CPU

#### **Why `utils.reset()`?**
- Clears internal neuron states (membrane potentials)
- Ensures NIR represents the network in clean initial state
- Does NOT affect trained weights/parameters

#### **Why `ignore_dims=[0]`?**
- Ignores batch dimension during tracing
- NIR format expects data shape without batch dimension
- Hardware processes single samples, not batches

## 📊 Usage Example

```python
# Train the network
train(num_epochs=1, lr=1e-2, log_every=50)

# Evaluate accuracy
test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"Final accuracy: {test_acc*100:.2f}%")

# Export to NIR
net.eval()
utils.reset(net)
net_cpu = copy.deepcopy(net).to("cpu")
utils.reset(net_cpu)

sample_cpu = torch.randn(1, 1, 28, 28)
nir_graph = export_to_nir(net_cpu, sample_cpu, ignore_dims=[0])
nir.write("csnn_mnist.nir", nir_graph)
```

## 🔄 Round-trip Validation

The project includes functionality to validate NIR export/import:

```python
# Load NIR back to PyTorch
from snntorch.import_nir import import_from_nir

nir_graph_loaded = nir.read("csnn_mnist.nir")
net_import = import_from_nir(nir_graph_loaded).to(device)

# Test imported network
test_acc_import = batch_accuracy_import(test_loader, net_import, num_steps)
print(f"Imported NIR accuracy: {test_acc_import*100:.2f}%")
```

## 📁 Project Structure

```
export_nir_from_snnTorch/
├── export_nir_from_snnTorchNet.py   # Main implementation with NIR export
├── export&import_nir.py             # Round-trip validation example
└── preview/                         # Development versions
    ├── v1.py
    ├── v2.py
    ├── v3.py
    └── readnir.py
```

## 🎯 Results

- **Training**: Achieves ~95%+ accuracy on MNIST in 1 epoch
- **NIR Export**: Successfully preserves trained weights and network structure
- **Round-trip**: Imported NIR models maintain comparable accuracy
- **Hardware Ready**: Generated `.nir` files can be deployed on neuromorphic platforms

## 🔬 Technical Notes

### **Memory Considerations**
- Training uses GPU/MPS when available for speed
- NIR export requires CPU copy due to NumPy conversion requirements
- Temporal simulation stores spike history for rate-based learning

### **Neuromorphic Deployment**
- NIR format enables cross-platform deployment
- Supports Intel Loihi, SpiNNaker, DYNAP-SE, and other neuromorphic chips
- Preserves temporal dynamics and spike-based computation

### **Extensibility**
- Architecture can be easily modified for different datasets
- LIF parameters (beta, threshold) can be made learnable
- Additional neuron models supported by extending NIR definitions

## 📚 References

- [snnTorch Tutorial 6](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
- [Neuromorphic Intermediate Representation (NIR)](https://neuroir.org/)
- [snnTorch Documentation](https://snntorch.readthedocs.io/)
- [NIRTorch GitHub](https://github.com/neuromorphs/nirtorch)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
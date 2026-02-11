import torch
from torch.utils.data import TensorDataset
import torch.serialization
import numpy as np

torch.serialization.add_safe_globals([TensorDataset])

ds_test = torch.load("nir_examples/rnn_test.pt")

X_test, y_test = ds_test.tensors

X_test = X_test.cpu()
y_test = y_test.cpu()

np.savetxt("inputs.txt", X_test.flatten(1).numpy(), fmt="%d")
np.savetxt("labels.txt", y_test.flatten().numpy(), fmt="%d")

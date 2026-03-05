import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from primitive_debug import run_cpp_impl

class SumPool2d(nn.Module):
    """
    SumPooling: soma todos os valores na janela (kernel)
    Similar ao AvgPool2d, mas sem dividir pelo tamanho da janela
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def forward(self, x):
        # x shape: [batch, channels, height, width] ou [channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Adiciona dimensão batch
            
        batch, channels, h, w = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Aplica padding se necessário
        if p_h > 0 or p_w > 0:
            x = nn.functional.pad(x, (p_w, p_w, p_h, p_h), mode='constant', value=0)
            h, w = x.shape[2], x.shape[3]
        
        # Calcula dimensões de saída
        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1
        
        # Cria tensor de saída
        output = torch.zeros(batch, channels, out_h, out_w)
        
        # Aplica sum pooling
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extrai a janela
                        start_h = i * s_h
                        start_w = j * s_w
                        window = x[b, c, start_h:start_h+k_h, start_w:start_w+k_w]
                        # Soma todos os valores
                        output[b, c, i, j] = window.sum()
        
        return output.squeeze(0) if batch == 1 else output


# Teste 1: Caso simples - 4x4 input, 2x2 kernel, stride 2, sem padding
print("=" * 60)
print("Teste 1: Input 4x4, Kernel 2x2, Stride 2, Padding 0")
print("=" * 60)

input_1 = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [9.0, 10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0, 16.0]]
])

sum_pool = SumPool2d(kernel_size=2, stride=2, padding=0)
output_1 = sum_pool(input_1)

print("\nInput shape:", input_1.shape)
print("Input:\n", input_1[0])
print("\nOutput shape:", output_1.shape)
print("Output:\n", output_1[0])
print("\nCálculo esperado:")
print(f"  [0,0] = {1+2+5+6} (soma de 1,2,5,6)")
print(f"  [0,1] = {3+4+7+8} (soma de 3,4,7,8)")
print(f"  [1,0] = {9+10+13+14} (soma de 9,10,13,14)")
print(f"  [1,1] = {11+12+15+16} (soma de 11,12,15,16)")


# Teste 2: Com múltiplos canais
print("\n" + "=" * 60)
print("Teste 2: Input 2 canais 3x3, Kernel 2x2, Stride 1, Padding 0")
print("=" * 60)

input_2 = torch.tensor([
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],
    
    [[0.5, 1.5, 2.5],
     [3.5, 4.5, 5.5],
     [6.5, 7.5, 8.5]]
])

sum_pool_2 = SumPool2d(kernel_size=2, stride=1, padding=0)
output_2 = sum_pool_2(input_2)

print("\nInput shape:", input_2.shape)
print("Canal 0:\n", input_2[0])
print("Canal 1:\n", input_2[1])
print("\nOutput shape:", output_2.shape)
print("Canal 0 Output:\n", output_2[0])
print("Canal 1 Output:\n", output_2[1])


# Teste 3: Com padding
print("\n" + "=" * 60)
print("Teste 3: Input 3x3, Kernel 2x2, Stride 1, Padding 1")
print("=" * 60)

input_3 = torch.tensor([
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]]
])

sum_pool_3 = SumPool2d(kernel_size=2, stride=1, padding=1)
output_3 = sum_pool_3(input_3)

print("\nInput shape:", input_3.shape)
print("Input:\n", input_3[0])
print("\nOutput shape:", output_3.shape)
print("Output:\n", output_3[0])


# Salva resultados em arquivo
with open("primitive_debug/SumPooling/out_py.txt", "w") as f:
    f.write("Teste 1: 4x4 -> 2x2\n")
    f.write(str(output_1[0].numpy()) + "\n\n")
    
    f.write("Teste 2: 2ch 3x3 -> 2x2\n")
    f.write("Canal 0:\n")
    f.write(str(output_2[0].numpy()) + "\n")
    f.write("Canal 1:\n")
    f.write(str(output_2[1].numpy()) + "\n\n")
    
    f.write("Teste 3: 3x3 com padding -> 4x4\n")
    f.write(str(output_3[0].numpy()) + "\n")

print("\n" + "=" * 60)
print("Resultados salvos em: primitive_debug/SumPooling/out_py.txt")
print("=" * 60)

# Executa implementação C++
run_cpp_impl()

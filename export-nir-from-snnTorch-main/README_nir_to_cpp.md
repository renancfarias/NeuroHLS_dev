# Parser NIR para C++

Este projeto implementa um parser que converte gráficos NIR (Neural Intermediate Representation) exportados do snnTorch em código C++ com funções vazias correspondentes às primitivas da rede neural.

## Estrutura do Projeto

```
export_nir_from_snnTorch/
├── nir_to_c.py              # Parser principal NIR → C++
├── export_nir_from_snnTorchNet.py  # Geração do arquivo NIR
├── preview/
│   └── readnir.py           # Visualização do gráfico NIR
└── cpp_output/              # Código C++ gerado
    ├── nir_network.h        # Header com declarações
    ├── nir_network.cpp      # Implementações vazias
    └── main.cpp             # Programa principal
```

## Como Usar

### 1. Gerar o arquivo NIR

Primeiro, execute o script que treina a rede e exporta para NIR:

```bash
python export_nir_from_snnTorchNet.py
```

Isso criará o arquivo `csnn_mnist.nir` contendo a representação da rede neural spiking.

### 2. Visualizar o gráfico NIR (opcional)

Para inspecionar a estrutura do gráfico:

```bash
python preview/readnir.py
```

### 3. Gerar código C++

Execute o parser para converter NIR em C++:

```bash
python nir_to_c.py
```

Isso criará os arquivos C++ em `cpp_output/`:
- `nir_network.h`: Declarações das funções
- `nir_network.cpp`: Implementações vazias com comentários
- `main.cpp`: Programa principal de exemplo

### 4. Compilar e testar

```bash
cd cpp_output
g++ -o test_network main.cpp nir_network.cpp
./test_network
```

## Arquitetura da Rede

A rede gerada segue a arquitetura CSNN (Convolutional Spiking Neural Network):

```
Input (1×28×28)
    ↓
Conv2D (1→12, kernel=5×5)
    ↓
AvgPool2D (kernel=2×2)  
    ↓
LIF Neurons (12×12×12)
    ↓
Conv2D (12→64, kernel=5×5)
    ↓
AvgPool2D (kernel=2×2)
    ↓
LIF Neurons (64×4×4)
    ↓
Flatten (64×4×4 → 1024)
    ↓
Linear/Affine (1024→10)
    ↓
LIF Neurons (10)
    ↓
Output (10 classes)
```

## Primitivas Implementadas

O parser mapeia os seguintes tipos NIR para funções C++:

| Tipo NIR     | Função C++         | Descrição                          |
|--------------|-------------------|------------------------------------|
| `Conv2d`     | `conv2d_layer()`  | Convolução 2D                     |
| `AvgPool2d`  | `avgpool2d_layer()` | Average Pooling 2D              |
| `LIF`        | `lif_neuron_layer()` | Neurônios Leaky Integrate-Fire  |
| `Affine`     | `linear_layer()`  | Camada totalmente conectada        |
| `Flatten`    | `flatten_layer()` | Achatar dimensões                  |

## Próximos Passos

As funções geradas são apenas stubs vazios. Para uma implementação completa, você precisará:

1. **Implementar as primitivas**: Adicionar a lógica de computação para cada tipo de camada
2. **Gerenciar dados**: Implementar estruturas para armazenar tensores e pesos
3. **Carregar parâmetros**: Extrair pesos e parâmetros do arquivo NIR e carregá-los no C++
4. **Otimizar**: Adicionar otimizações específicas do hardware

## Parâmetros Extraídos

O parser extrai automaticamente os seguintes parâmetros dos nós NIR:

- **Conv2d**: `weight_shape`, `bias_shape`, `stride`, `padding`, `input_shape`
- **AvgPool2d**: `kernel_size`, `stride`, `padding`
- **LIF**: `tau_shape`, `r_shape`, `v_threshold_shape`, `v_leak_shape`, `v_reset_shape`
- **Affine**: `weight_shape`, `bias_shape`
- **Flatten**: `start_dim`, `end_dim`

Estes parâmetros são incluídos como comentários no código C++ gerado.

## Dependências

- Python 3.7+
- PyTorch
- snnTorch
- NIR (`pip install nir`)
- NumPy

Para compilar o código C++:
- GCC ou Clang com suporte C++11 ou superior

## Exemplo de Saída

```cpp
void execute_network() {
    std::cout << "=== Executando Rede Neural Spiking ===" << std::endl;

    // Node 0: Conv2d
    // weight_shape: [12, 1, 5, 5]
    // bias_shape: [12]
    // stride: [1, 1]
    // padding: [0, 0]
    conv2d_layer(); // Layer 0

    // ... outras camadas ...
    
    std::cout << "=== Execução concluída ===" << std::endl;
}
```

## Contribuindo

Para adicionar suporte a novas primitivas NIR:

1. Adicione o mapeamento em `function_mapping` no método `_generate_function_call()`
2. Adicione a declaração da função no método `generate_cpp_header()`  
3. Adicione a implementação vazia no método `generate_cpp_implementation()`
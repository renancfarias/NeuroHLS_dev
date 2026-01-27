from typing import List, Union, Tuple, Optional, Any
import numpy as np
import nir

NUM_DASHES = 50

class LayerConfig:

    def __init__(self):

        self.is_recurrent = False
        self.dependencies = []

    def add_dependency(self, name: str, is_recurrent: bool):
        self.dependencies.append((name, is_recurrent))

    def __str__(self):
        s = f"\tIs recurrent: {'YES' if self.is_recurrent else 'NO'}\n"
        s += "\tDependencies:\n"

        for (name, is_recurrent) in self.dependencies:
            s += f"\t   - {name} ({'recurrent' if is_recurrent else 'ready'})\n"

        return s
    
class Input(LayerConfig):
    
    def __init__(self, input_shape):

        super().__init__()
        self.input_shape = input_shape

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Input ({self.input_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += super().__str__()
        return s
    
class Output(LayerConfig):
    
    def __init__(self, output_shape):

        super().__init__()
        self.output_shape = output_shape

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Output ({self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += super().__str__()
        return s
    
class Affine(LayerConfig):
    
    def __init__(self, n_inputs: int, n_neurons: int):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Affine ({self.n_inputs}, {self.n_neurons})\n"
        s += "-" * NUM_DASHES + "\n"
        
        s += super().__str__()
        return s

class Flatten(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple, start_dim: int = 1, end_dim: int = -1):
        """
        Flatten layer configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            start_dim: First dimension to flatten (default: 1)
            end_dim: Last dimension to flatten (default: -1, meaning last dimension)
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Flatten (input: {self.input_shape}, output: {self.output_shape})\n"
        # s += f"\tstart_dim={self.start_dim}, end_dim={self.end_dim}\n"
        s += "-" * NUM_DASHES + "\n"
        
        s += super().__str__()
        return s

class Conv1d(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple, weight: np.ndarray, 
                 stride: int, padding: Union[int, str], dilation: int, 
                 groups: int, bias: np.ndarray):
        """
        Conv1d layer configuration.
        
        Args:
            input_shape: Shape of the input tensor (C_in, N)
            output_shape: Shape of the output tensor (C_out, N_out) - calculated by NIR
            weight: Weight array with shape (C_out, C_in, N)
            stride: Stride
            padding: Padding (int or 'same'/'valid')
            dilation: Dilation
            groups: Groups
            bias: Bias array of shape (C_out,)
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.weight = weight
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Conv1d (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        s += f"\tStride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}\n"
        s += f"\tGroups: {self.groups}, Bias shape: {self.bias.shape if self.bias is not None else None}\n"
        
        s += super().__str__()
        return s

class Conv2d(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple, weight: np.ndarray,
                 stride: Union[int, Tuple[int, int]], 
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]], 
                 groups: int, bias: np.ndarray):
        """
        Conv2d layer configuration.
        
        Args:
            input_shape: Shape of the input tensor (C_in, N_x, N_y)
            output_shape: Shape of the output tensor (C_out, N_x_out, N_y_out) - calculated by NIR
            weight: Weight array with shape (C_out, C_in, W_x, W_y)
            stride: Stride (int or tuple)
            padding: Padding (int, tuple, or 'same'/'valid')
            dilation: Dilation (int or tuple)
            groups: Groups
            bias: Bias array of shape (C_out,)
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.weight = weight
        
        # Normalize stride, padding, dilation to tuples if they are ints
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) and not isinstance(padding, str) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        
        self.groups = groups
        self.bias = bias
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Conv2d (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        s += f"\tStride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}\n"
        s += f"\tGroups: {self.groups}, Bias shape: {self.bias.shape if self.bias is not None else None}\n"

        s += super().__str__()
        return s

class CubaLI(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple, 
                 tau_syn: np.ndarray, tau_mem: np.ndarray, 
                 r: np.ndarray, v_leak: np.ndarray, w_in: np.ndarray):
        """
        Current based leaky integrator model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            tau_syn: Synaptic time constant array
            tau_mem: Membrane time constant array
            r: Resistance array
            v_leak: Leak voltage array
            w_in: Input current weight array
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.r = r
        self.v_leak = v_leak
        self.w_in = w_in
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"CubaLI (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shapes: {self.tau_syn.shape}\n"
        s += f"\ttau_syn range: [{self.tau_syn.min():.4f}, {self.tau_syn.max():.4f}]\n"
        s += f"\ttau_mem range: [{self.tau_mem.min():.4f}, {self.tau_mem.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += f"\tw_in range: [{self.w_in.min():.4f}, {self.w_in.max():.4f}]\n"
        
        s += super().__str__()
        return s

class CubaLIF(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 tau_syn: np.ndarray, tau_mem: np.ndarray,
                 r: np.ndarray, v_leak: np.ndarray,
                 v_threshold: np.ndarray, v_reset: np.ndarray, w_in: np.ndarray):
        """
        Current based leaky integrate-and-fire neuron model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            tau_syn: Synaptic time constant array
            tau_mem: Membrane time constant array
            r: Resistance array
            v_leak: Leak voltage array
            v_threshold: Firing threshold array
            v_reset: Reset potential array
            w_in: Input current weight array
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.r = r
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.w_in = w_in
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"CubaLIF (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shapes: {self.tau_syn.shape}\n"
        s += f"\ttau_syn range: [{self.tau_syn.min():.4f}, {self.tau_syn.max():.4f}]\n"
        s += f"\ttau_mem range: [{self.tau_mem.min():.4f}, {self.tau_mem.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        s += f"\tw_in range: [{self.w_in.min():.4f}, {self.w_in.max():.4f}]\n"
        
        s += super().__str__()
        return s

class I(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple, r: np.ndarray):
        """
        Integrator neuron model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            r: Resistance array
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.r = r
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Integrator (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.r.shape}\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        
        s += super().__str__()
        return s

class IF(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 r: np.ndarray, v_threshold: np.ndarray, v_reset: np.ndarray):
        """
        Integrate-and-fire neuron model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            r: Resistance array
            v_threshold: Firing threshold array
            v_reset: Reset potential array
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.r = r
        self.v_threshold = v_threshold
        self.v_reset = v_reset
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"IF (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.r.shape}\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        
        s += super().__str__()
        return s

class LI(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 tau: np.ndarray, r: np.ndarray, v_leak: np.ndarray):
        """
        Leaky integrator neuron model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            tau: Time constant array
            r: Resistance array
            v_leak: Leak voltage array
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau = tau
        self.r = r
        self.v_leak = v_leak
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"LI (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.tau.shape}\n"
        s += f"\ttau range: [{self.tau.min():.4f}, {self.tau.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        
        s += super().__str__()
        return s

class LIF(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 tau: np.ndarray, r: np.ndarray, v_leak: np.ndarray,
                 v_threshold: np.ndarray, v_reset: np.ndarray):
        """
        Leaky integrate-and-fire neuron model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            tau: Time constant array
            r: Resistance array
            v_leak: Leak voltage array
            v_threshold: Firing threshold array
            v_reset: Reset potential array
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau = tau
        self.r = r
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"LIF (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.tau.shape}\n"
        s += f"\ttau range: [{self.tau.min():.4f}, {self.tau.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        
        s += super().__str__()
        return s

class SumPool2d(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]]):
        """
        Sum pooling layer in 2d configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            kernel_size: Size of pooling kernel (Height, Width)
            stride: Stride (Height, Width)
            padding: Padding (Height, Width)
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        
        # Normalize to tuples if they are ints
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"SumPool2d (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        
        s += super().__str__()
        return s

class AvgPool2d(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple,
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]]):
        """
        Average pooling layer in 2d configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            kernel_size: Size of pooling kernel (Height, Width)
            stride: Stride (Height, Width)
            padding: Padding (Height, Width)
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        
        # Normalize to tuples if they are ints
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"AvgPool2d (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        
        s += super().__str__()
        return s

class Linear(LayerConfig):
    
    def __init__(self, input_shape: tuple, output_shape: tuple, weight: np.ndarray):
        """
        Linear transform without bias configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            weight: Weight matrix
        """
        super().__init__()
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.weight = weight
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Linear (input: {self.input_shape}, output: {self.output_shape})\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        
        s += super().__str__()
        return s
    
class ModelConfig:
    
    def __init__(self):

        self.layers: List[LayerConfig] = []

        self.input_total_bits = 16
        self.input_int_bits = 8

    def __str__(self):
        
        s = ""
        for idx, layer in enumerate(self.layers):

            if idx > 0:
                s += "\n"

            s += layer.__str__()
        
        return s
    
    def add_layer(self, layer: LayerConfig):

        self.layers.append(layer)

# def build_model_from_nir(nir_file: str) -> ModelConfig:
#     """
#     Constrói um ModelConfig a partir de um arquivo NIR usando busca em largura (BFS).
    
#     Args:
#         nir_file: Caminho para o arquivo .nir
        
#     Returns:
#         ModelConfig com todas as camadas configuradas
#     """
#     from collections import deque
    
#     # Lê o grafo NIR
#     nir_graph = nir.read(nir_file)
#     nodes = nir_graph.nodes
#     edges = nir_graph.edges
    
#     # Constrói o grafo de adjacências
#     graph = {}
#     for node_name in nodes.keys():
#         graph[node_name] = []
    
#     for src, dst in edges:
#         if src in graph:
#             graph[src].append(dst)
    
#     # Inicializa a busca em largura
#     visited = set()
#     queue = deque()
#     model_config = ModelConfig()
    
#     # Começa do nó 'input'
#     queue.append('input')
    
#     # Dicionário para guardar os shapes calculados de cada nó
#     shapes = {}
    
#     # Processa o nó de entrada
#     if 'input' in nodes:
#         input_node = nodes['input']
#         if hasattr(input_node, 'output_type') and 'output' in input_node.output_type:
#             input_shape = tuple(input_node.output_type['output'])
#         elif hasattr(input_node, 'input_type') and 'input' in input_node.input_type:
#             input_shape = tuple(input_node.input_type['input'])
#         else:
#             raise ValueError("Não foi possível determinar o shape de entrada do NIR")
        
#         shapes['input'] = input_shape
    
#     # BFS
#     while queue:
#         current_node_name = queue.popleft()
        
#         if current_node_name in visited:
#             continue
            
#         visited.add(current_node_name)
        
#         # Pega o nó atual
#         if current_node_name not in nodes:
#             continue
            
#         current_node = nodes[current_node_name]
        
#         # Determina o input_shape para este nó
#         # (vem do shape de saída do nó anterior)
#         if current_node_name == 'input':
#             current_input_shape = shapes['input']
#         else:
#             # Encontra o predecessor (assumindo que há apenas um por enquanto)
#             predecessors = [src for src, dst in edges if dst == current_node_name]
#             if predecessors:
#                 # Pega o shape do primeiro predecessor
#                 current_input_shape = shapes.get(predecessors[0], None)
#                 if current_input_shape is None:
#                     print(f"Aviso: Shape de entrada não encontrado para {current_node_name}")
#                     current_input_shape = (1,)  # placeholder
#             else:
#                 current_input_shape = (1,)  # placeholder
        
#         # Determina o output_shape
#         if hasattr(current_node, 'output_type') and 'output' in current_node.output_type:
#             current_output_shape = tuple(current_node.output_type['output'])
#         elif hasattr(current_node, 'input_type') and 'output' in current_node.input_type:
#             current_output_shape = tuple(current_node.input_type['output'])
#         else:
#             # Tenta inferir do próprio nó
#             current_output_shape = current_input_shape  # placeholder
        
#         # Salva o shape de saída deste nó
#         shapes[current_node_name] = current_output_shape
        
#         # Extrai informações do nó e cria a camada
#         try:
#             layer = get_info_from_node(current_node, current_input_shape, current_output_shape)
            
#             # Adiciona a camada ao modelo (se não for None)
#             if layer is not None:
#                 model_config.add_layer(layer)
#                 print(f"Camada adicionada: {current_node_name} ({type(current_node).__name__})")
        
#         except Exception as e:
#             print(f"Erro ao processar nó {current_node_name}: {e}")
        
#         # Adiciona os vizinhos à fila
#         for neighbor in graph.get(current_node_name, []):
#             if neighbor not in visited:
#                 queue.append(neighbor)
    
#     return model_config

# def create_model_config_from_nir(nir_graph) -> ModelConfig:
#     """
#     Constrói um ModelConfig a partir de um objeto NIR Graph.
    
#     Args:
#         nir_graph: Objeto NIRGraph já carregado
        
#     Returns:
#         ModelConfig com todas as camadas configuradas
#     """
#     from collections import deque
    
#     nodes = nir_graph.nodes
#     edges = nir_graph.edges
    
#     # Constrói o grafo de adjacências
#     graph = {}
#     for node_name in nodes.keys():
#         graph[node_name] = []
    
#     for src, dst in edges:
#         if src in graph:
#             graph[src].append(dst)
    
#     # Inicializa a busca em largura
#     visited = set()
#     queue = deque()
#     model_config = ModelConfig()
    
#     # Começa do nó 'input'
#     queue.append('input')
    
#     # Dicionário para guardar os shapes calculados de cada nó
#     shapes = {}
    
#     # Processa o nó de entrada
#     if 'input' in nodes:
#         input_node = nodes['input']
#         if hasattr(input_node, 'output_type') and 'output' in input_node.output_type:
#             input_shape = tuple(input_node.output_type['output'])
#         elif hasattr(input_node, 'input_type') and 'input' in input_node.input_type:
#             input_shape = tuple(input_node.input_type['input'])
#         else:
#             raise ValueError("Não foi possível determinar o shape de entrada do NIR")
        
#         shapes['input'] = input_shape
    
#     # BFS
#     while queue:
#         current_node_name = queue.popleft()
        
#         if current_node_name in visited:
#             continue
            
#         visited.add(current_node_name)
        
#         # Pega o nó atual
#         if current_node_name not in nodes:
#             continue
            
#         current_node = nodes[current_node_name]
        
#         # Determina o input_shape para este nó
#         if current_node_name == 'input':
#             current_input_shape = shapes['input']
#         else:
#             # Encontra o predecessor
#             predecessors = [src for src, dst in edges if dst == current_node_name]
#             if predecessors:
#                 current_input_shape = shapes.get(predecessors[0], None)
#                 if current_input_shape is None:
#                     print(f"Aviso: Shape de entrada não encontrado para {current_node_name}")
#                     current_input_shape = (1,)
#             else:
#                 current_input_shape = (1,)
        
#         # Determina o output_shape
#         if hasattr(current_node, 'output_type') and 'output' in current_node.output_type:
#             current_output_shape = tuple(current_node.output_type['output'])
#         elif hasattr(current_node, 'input_type') and 'output' in current_node.input_type:
#             current_output_shape = tuple(current_node.input_type['output'])
#         else:
#             current_output_shape = current_input_shape
        
#         # Salva o shape de saída deste nó
#         shapes[current_node_name] = current_output_shape
        
#         # Extrai informações do nó e cria a camada
#         try:
#             layer = get_info_from_node(current_node, current_input_shape, current_output_shape)
            
#             if layer is not None:
#                 model_config.add_layer(layer)
#                 print(f"Camada adicionada: {current_node_name} ({type(current_node).__name__})")
        
#         except Exception as e:
#             print(f"Erro ao processar nó {current_node_name}: {e}")
        
#         # Adiciona os vizinhos à fila
#         for neighbor in graph.get(current_node_name, []):
#             if neighbor not in visited:
#                 queue.append(neighbor)
    
#     return model_config

# def teste():
#     model = build_model_from_nir("z_nir_examples/lif_norse.nir")
#     print(model)

# teste()
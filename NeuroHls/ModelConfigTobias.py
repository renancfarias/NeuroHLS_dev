from typing import List, Union, Tuple, Optional, Any
import numpy as np
from .FileGenUtils import get_closest_divisor
import nir

class LayerConfig:

    def __init__(self):

        self._accum_unroll_factor = 1
        self._fire_unroll_factor = 1

        self._potentials_total_bits = 16
        self._potentials_int_bits = 8

    def __str__(self):
        s = "\tUnroll Factors:\n"
        s += f"\t\t- Accum: {self._accum_unroll_factor}\n"
        s += f"\t\t- Fire: {self._fire_unroll_factor}\n"

        s += f"\tQuantization (ap_fixed<{self._potentials_total_bits}, {self._potentials_int_bits}>):\n"
        s += f"\t\t- Total bits: {self._potentials_total_bits}\n"
        s += f"\t\t- Integer bits: {self._potentials_int_bits}\n"
        s += f"\t\t- Fractional bits: {self._potentials_total_bits - self._potentials_int_bits}\n"

        return s
    
    def set_potential_quantization(self, total_bits, int_bits):
        self._potentials_total_bits = total_bits
        self._potentials_int_bits = min(int_bits, total_bits)

    def set_unroll_factors(self, accum_unroll_factor, fire_unroll_factor):
        self._accum_unroll_factor = accum_unroll_factor
        self._fire_unroll_factor = fire_unroll_factor

    def get_quantization(self):

        return (self._potentials_total_bits, self._potentials_int_bits)

    def get_accum_unroll_factor(self):
        
        return self._accum_unroll_factor
    
    def get_fire_unroll_factor(self):

        return self._fire_unroll_factor
    
class Affine(LayerConfig):
    
    def __init__(self, n_inputs: int, n_neurons: int):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def __str__(self):
        s = f"Afine ({self.n_inputs}, {self.n_neurons})\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()

        return s
    
    def get_input_shape(self):

        return np.array([self.n_inputs])
    
    def get_output_shape(self):
        
        return np.array([self.n_neurons])

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
        s = f"Flatten (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tstart_dim={self.start_dim}, end_dim={self.end_dim}\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"Conv1d (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tWeight shape: {self.weight.shape}\n"
        s += f"\tStride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}\n"
        s += f"\tGroups: {self.groups}, Bias shape: {self.bias.shape if self.bias is not None else None}\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"Conv2d (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tWeight shape: {self.weight.shape}\n"
        s += f"\tStride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}\n"
        s += f"\tGroups: {self.groups}, Bias shape: {self.bias.shape if self.bias is not None else None}\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"CubaLI (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tParameter shapes: {self.tau_syn.shape}\n"
        s += f"\ttau_syn range: [{self.tau_syn.min():.4f}, {self.tau_syn.max():.4f}]\n"
        s += f"\ttau_mem range: [{self.tau_mem.min():.4f}, {self.tau_mem.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += f"\tw_in range: [{self.w_in.min():.4f}, {self.w_in.max():.4f}]\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"CubaLIF (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tParameter shapes: {self.tau_syn.shape}\n"
        s += f"\ttau_syn range: [{self.tau_syn.min():.4f}, {self.tau_syn.max():.4f}]\n"
        s += f"\ttau_mem range: [{self.tau_mem.min():.4f}, {self.tau_mem.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        s += f"\tw_in range: [{self.w_in.min():.4f}, {self.w_in.max():.4f}]\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"Integrator (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tParameter shape: {self.r.shape}\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"IF (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tParameter shape: {self.r.shape}\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"LI (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tParameter shape: {self.tau.shape}\n"
        s += f"\ttau range: [{self.tau.min():.4f}, {self.tau.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"LIF (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tParameter shape: {self.tau.shape}\n"
        s += f"\ttau range: [{self.tau.min():.4f}, {self.tau.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"SumPool2d (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"AvgPool2d (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

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
        s = f"Linear (input: {tuple(self.input_shape)}, output: {tuple(self.output_shape)})\n"
        s += f"\tWeight shape: {self.weight.shape}\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()
        
        return s
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape

class DenseLayerConfig(LayerConfig):

    def __init__(self, n_inputs: int, n_neurons: int):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def __str__(self):
        s = f"Dense ({self.n_inputs}, {self.n_neurons})\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()

        return s
    
    def set_unroll_factors(self, accum_unroll_factor, fire_unroll_factor, verbose=True):

        correct_accum_unroll_factor = get_closest_divisor(self.n_inputs, accum_unroll_factor)
        correct_fire_unroll_factor = get_closest_divisor(self.n_neurons, fire_unroll_factor)

        super().set_unroll_factors(correct_accum_unroll_factor, correct_fire_unroll_factor)

        if not verbose:
            return

        if correct_accum_unroll_factor != accum_unroll_factor:
            print(f"The accumulation unroll factor must be a divisor of the number of inputs ({self.n_inputs}). Used value: {correct_accum_unroll_factor}")
        else:
            print(f"Used accumulation unroll factor: {correct_accum_unroll_factor}")

        if correct_fire_unroll_factor != fire_unroll_factor:
            print(f"The fire unroll factor must be a divisor of the number of neurons ({self.n_neurons}). Used value: {correct_fire_unroll_factor}")
        else:
            print(f"Used fire unroll factor: {correct_fire_unroll_factor}")

    def get_input_shape(self):

        return (self.n_inputs, )
    
    def get_output_shape(self):
        
        return (self.n_neurons, )
    
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

            s += 30 * "-" + "\n"
            s += f"Layer {idx+1}: {layer.__str__()}"
        
        return s
    
    def add_layer(self, layer: LayerConfig):

        self.layers.append(layer)


def get_info_from_node(node: Any, input_shape: tuple, output_shape: tuple) -> Optional[LayerConfig]:
    """
    Extrai informações de um nó NIR e retorna a instância da classe apropriada.
    
    Args:
        node: Nó do grafo NIR
        input_shape: Shape da entrada para esta camada
        output_shape: Shape da saída desta camada
        
    Returns:
        Instância de LayerConfig apropriada ou None se o tipo não for reconhecido
    """
    node_type = type(node).__name__
    
    # Affine layer (camada linear com bias)
    if node_type == 'Affine':
        # Affine tem weight de shape [out, in] e bias de shape [out]
        if hasattr(node, 'weight') and node.weight is not None:
            n_neurons = node.weight.shape[0]
            n_inputs = node.weight.shape[1]
            return Affine(n_inputs, n_neurons)
        else:
            raise ValueError(f"Nó Affine sem weight: {node}")
    
    # Flatten layer
    elif node_type == 'Flatten':
        start_dim = getattr(node, 'start_dim', 1)
        end_dim = getattr(node, 'end_dim', -1)
        return Flatten(input_shape, output_shape, start_dim, end_dim)
    
    # Conv1d layer
    elif node_type == 'Conv1d':
        weight = node.weight
        stride = getattr(node, 'stride', 1)
        padding = getattr(node, 'padding', 0)
        dilation = getattr(node, 'dilation', 1)
        groups = getattr(node, 'groups', 1)
        bias = getattr(node, 'bias', None)
        
        return Conv1d(input_shape, output_shape, weight, stride, padding, dilation, groups, bias)
    
    # Conv2d layer
    elif node_type == 'Conv2d':
        weight = node.weight
        stride = getattr(node, 'stride', 1)
        padding = getattr(node, 'padding', 0)
        dilation = getattr(node, 'dilation', 1)
        groups = getattr(node, 'groups', 1)
        bias = getattr(node, 'bias', None)
        
        return Conv2d(input_shape, output_shape, weight, stride, padding, dilation, groups, bias)
    
    # CubaLI (Current-based Leaky Integrator)
    elif node_type == 'CubaLI':
        tau_syn = node.tau_syn
        tau_mem = node.tau_mem
        r = node.r
        v_leak = node.v_leak
        w_in = node.w_in
        
        return CubaLI(input_shape, output_shape, tau_syn, tau_mem, r, v_leak, w_in)
    
    # CubaLIF (Current-based Leaky Integrate-and-Fire)
    elif node_type == 'CubaLIF':
        tau_syn = node.tau_syn
        tau_mem = node.tau_mem
        r = node.r
        v_leak = node.v_leak
        v_threshold = node.v_threshold
        v_reset = node.v_reset
        w_in = node.w_in
        
        return CubaLIF(input_shape, output_shape, tau_syn, tau_mem, r, v_leak, v_threshold, v_reset, w_in)
    
    # I (Integrator)
    elif node_type == 'I':
        r = node.r
        return I(input_shape, output_shape, r)
    
    # IF (Integrate-and-Fire)
    elif node_type == 'IF':
        r = node.r
        v_threshold = node.v_threshold
        v_reset = node.v_reset
        
        return IF(input_shape, output_shape, r, v_threshold, v_reset)
    
    # LI (Leaky Integrator)
    elif node_type == 'LI':
        tau = node.tau
        r = node.r
        v_leak = node.v_leak
        
        return LI(input_shape, output_shape, tau, r, v_leak)
    
    # LIF (Leaky Integrate-and-Fire)
    elif node_type == 'LIF':
        tau = node.tau
        r = node.r
        v_leak = node.v_leak
        v_threshold = node.v_threshold
        v_reset = node.v_reset
        
        return LIF(input_shape, output_shape, tau, r, v_leak, v_threshold, v_reset)
    
    # SumPool2d
    elif node_type == 'SumPool2d':
        kernel_size = getattr(node, 'kernel_size', 2)
        stride = getattr(node, 'stride', None)
        if stride is None:
            stride = kernel_size
        padding = getattr(node, 'padding', 0)
        
        return SumPool2d(input_shape, output_shape, kernel_size, stride, padding)
    
    # AvgPool2d
    elif node_type == 'AvgPool2d':
        kernel_size = getattr(node, 'kernel_size', 2)
        stride = getattr(node, 'stride', None)
        if stride is None:
            stride = kernel_size
        padding = getattr(node, 'padding', 0)
        
        return AvgPool2d(input_shape, output_shape, kernel_size, stride, padding)
    
    # Linear (sem bias)
    elif node_type == 'Linear':
        weight = node.weight
        return Linear(input_shape, output_shape, weight)
    
    # Tipos especiais que não geram camadas
    elif node_type in ['Input', 'Output']:
        return None
    
    # Tipo não reconhecido
    else:
        raise ValueError(f"Tipo de nó não reconhecido: {node_type}")

def build_model_from_nir(nir_file: str) -> ModelConfig:
    """
    Constrói um ModelConfig a partir de um arquivo NIR usando busca em largura (BFS).
    
    Args:
        nir_file: Caminho para o arquivo .nir
        
    Returns:
        ModelConfig com todas as camadas configuradas
    """
    from collections import deque
    
    # Lê o grafo NIR
    nir_graph = nir.read(nir_file)
    nodes = nir_graph.nodes
    edges = nir_graph.edges
    
    # Constrói o grafo de adjacências
    graph = {}
    for node_name in nodes.keys():
        graph[node_name] = []
    
    for src, dst in edges:
        if src in graph:
            graph[src].append(dst)
    
    # Inicializa a busca em largura
    visited = set()
    queue = deque()
    model_config = ModelConfig()
    
    # Começa do nó 'input'
    queue.append('input')
    
    # Dicionário para guardar os shapes calculados de cada nó
    shapes = {}
    
    # Processa o nó de entrada
    if 'input' in nodes:
        input_node = nodes['input']
        if hasattr(input_node, 'output_type') and 'output' in input_node.output_type:
            input_shape = tuple(input_node.output_type['output'])
        elif hasattr(input_node, 'input_type') and 'input' in input_node.input_type:
            input_shape = tuple(input_node.input_type['input'])
        else:
            raise ValueError("Não foi possível determinar o shape de entrada do NIR")
        
        shapes['input'] = input_shape
    
    # BFS
    while queue:
        current_node_name = queue.popleft()
        
        if current_node_name in visited:
            continue
            
        visited.add(current_node_name)
        
        # Pega o nó atual
        if current_node_name not in nodes:
            continue
            
        current_node = nodes[current_node_name]
        
        # Determina o input_shape para este nó
        # (vem do shape de saída do nó anterior)
        if current_node_name == 'input':
            current_input_shape = shapes['input']
        else:
            # Encontra o predecessor (assumindo que há apenas um por enquanto)
            predecessors = [src for src, dst in edges if dst == current_node_name]
            if predecessors:
                # Pega o shape do primeiro predecessor
                current_input_shape = shapes.get(predecessors[0], None)
                if current_input_shape is None:
                    print(f"Aviso: Shape de entrada não encontrado para {current_node_name}")
                    current_input_shape = (1,)  # placeholder
            else:
                current_input_shape = (1,)  # placeholder
        
        # Determina o output_shape
        if hasattr(current_node, 'output_type') and 'output' in current_node.output_type:
            current_output_shape = tuple(current_node.output_type['output'])
        elif hasattr(current_node, 'input_type') and 'output' in current_node.input_type:
            current_output_shape = tuple(current_node.input_type['output'])
        else:
            # Tenta inferir do próprio nó
            current_output_shape = current_input_shape  # placeholder
        
        # Salva o shape de saída deste nó
        shapes[current_node_name] = current_output_shape
        
        # Extrai informações do nó e cria a camada
        try:
            layer = get_info_from_node(current_node, current_input_shape, current_output_shape)
            
            # Adiciona a camada ao modelo (se não for None)
            if layer is not None:
                model_config.add_layer(layer)
                print(f"Camada adicionada: {current_node_name} ({type(current_node).__name__})")
        
        except Exception as e:
            print(f"Erro ao processar nó {current_node_name}: {e}")
        
        # Adiciona os vizinhos à fila
        for neighbor in graph.get(current_node_name, []):
            if neighbor not in visited:
                queue.append(neighbor)
    
    return model_config

def create_model_config_from_nir(nir_graph) -> ModelConfig:
    """
    Constrói um ModelConfig a partir de um objeto NIR Graph.
    
    Args:
        nir_graph: Objeto NIRGraph já carregado
        
    Returns:
        ModelConfig com todas as camadas configuradas
    """
    from collections import deque
    
    nodes = nir_graph.nodes
    edges = nir_graph.edges
    
    # Constrói o grafo de adjacências
    graph = {}
    for node_name in nodes.keys():
        graph[node_name] = []
    
    for src, dst in edges:
        if src in graph:
            graph[src].append(dst)
    
    # Inicializa a busca em largura
    visited = set()
    queue = deque()
    model_config = ModelConfig()
    
    # Começa do nó 'input'
    queue.append('input')
    
    # Dicionário para guardar os shapes calculados de cada nó
    shapes = {}
    
    # Processa o nó de entrada
    if 'input' in nodes:
        input_node = nodes['input']
        if hasattr(input_node, 'output_type') and 'output' in input_node.output_type:
            input_shape = tuple(input_node.output_type['output'])
        elif hasattr(input_node, 'input_type') and 'input' in input_node.input_type:
            input_shape = tuple(input_node.input_type['input'])
        else:
            raise ValueError("Não foi possível determinar o shape de entrada do NIR")
        
        shapes['input'] = input_shape
    
    # BFS
    while queue:
        current_node_name = queue.popleft()
        
        if current_node_name in visited:
            continue
            
        visited.add(current_node_name)
        
        # Pega o nó atual
        if current_node_name not in nodes:
            continue
            
        current_node = nodes[current_node_name]
        
        # Determina o input_shape para este nó
        if current_node_name == 'input':
            current_input_shape = shapes['input']
        else:
            # Encontra o predecessor
            predecessors = [src for src, dst in edges if dst == current_node_name]
            if predecessors:
                current_input_shape = shapes.get(predecessors[0], None)
                if current_input_shape is None:
                    print(f"Aviso: Shape de entrada não encontrado para {current_node_name}")
                    current_input_shape = (1,)
            else:
                current_input_shape = (1,)
        
        # Determina o output_shape
        if hasattr(current_node, 'output_type') and 'output' in current_node.output_type:
            current_output_shape = tuple(current_node.output_type['output'])
        elif hasattr(current_node, 'input_type') and 'output' in current_node.input_type:
            current_output_shape = tuple(current_node.input_type['output'])
        else:
            current_output_shape = current_input_shape
        
        # Salva o shape de saída deste nó
        shapes[current_node_name] = current_output_shape
        
        # Extrai informações do nó e cria a camada
        try:
            layer = get_info_from_node(current_node, current_input_shape, current_output_shape)
            
            if layer is not None:
                model_config.add_layer(layer)
                print(f"Camada adicionada: {current_node_name} ({type(current_node).__name__})")
        
        except Exception as e:
            print(f"Erro ao processar nó {current_node_name}: {e}")
        
        # Adiciona os vizinhos à fila
        for neighbor in graph.get(current_node_name, []):
            if neighbor not in visited:
                queue.append(neighbor)
    
    return model_config
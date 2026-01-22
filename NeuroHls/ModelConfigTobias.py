from typing import List, Union, Tuple, Optional
import numpy as np
from .FileGenUtils import get_closest_divisor

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

    
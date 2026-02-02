from typing import List, Union, Tuple, Optional, Any
import numpy as np
import nir

NUM_DASHES = 55

class LayerConfig:

    def __init__(self, name):

        self.is_recurrent = False
        self.dependencies = []
        self.name = name
        self.emits_spike = False

    def add_dependency(self, name: str, is_recurrent: bool):
        self.dependencies.append((name, is_recurrent))

    def __str__(self):
        s = f"\tIs recurrent: {'YES' if self.is_recurrent else 'NO'}\n"
        s += "\tDependencies:\n"

        for (name, is_recurrent) in self.dependencies:
            s += f"\t   - {name} ({'recurrent' if is_recurrent else 'ready'})\n"

        return s
    
class Merge(LayerConfig):

    def __init__(self, name: str, layer_1: str, layer_2: str, shape):
        super().__init__(name)
        
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.input_shape = shape
        self.output_shape = shape

    def __str__(self):
        s = NUM_DASHES * "-" + "\n"
        s += f"Merge (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += NUM_DASHES * "-" + "\n"

        s += f"\tLayer 1: {self.layer_1}\n"
        s += f"\tLayer 2: {self.layer_2}\n"

        return s + super().__str__()
    
class Input(LayerConfig):
    
    def __init__(self, name: str, input_shape):

        super().__init__(name)
        self.input_shape = input_shape

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Input ({self.input_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += super().__str__()
        return s
    
class Output(LayerConfig):
    
    def __init__(self, name: str, output_shape):

        super().__init__(name)
        self.output_shape = output_shape
        self.emits_spike = True

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Output ({self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += super().__str__()
        return s
    
class Affine(LayerConfig):
    
    def __init__(self, name: str, input_shape, output_shape):

        super().__init__(name)
        self.input_shape = np.atleast_1d(input_shape)
        self.output_shape = np.atleast_1d(output_shape)

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Affine (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"
        
        s += super().__str__()
        return s

class Flatten(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, start_dim: int = 1, end_dim: int = -1):
        """
        Flatten layer configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            start_dim: First dimension to flatten (default: 1)
            end_dim: Last dimension to flatten (default: -1, meaning last dimension)
        """
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Flatten (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        # s += f"\tstart_dim={self.start_dim}, end_dim={self.end_dim}\n"
        s += "-" * NUM_DASHES + "\n"
        
        s += super().__str__()
        return s

class Conv1d(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, weight: np.ndarray, 
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
        super().__init__(name)
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
        s += f"Conv1d (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        s += f"\tStride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}\n"
        s += f"\tGroups: {self.groups}, Bias shape: {self.bias.shape if self.bias is not None else None}\n"
        
        s += super().__str__()
        return s

class Conv2d(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, weight: np.ndarray,
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
        super().__init__(name)
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
        s += f"Conv2d (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        s += f"\tStride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}\n"
        s += f"\tGroups: {self.groups}, Bias shape: {self.bias.shape if self.bias is not None else None}\n"

        s += super().__str__()
        return s

class CubaLI(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, 
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.r = r
        self.v_leak = v_leak
        self.w_in = w_in
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"CubaLI (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
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
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple,
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.r = r
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.w_in = w_in
        self.emits_spike = True
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"CubaLIF (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
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
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, r: np.ndarray):
        """
        Integrator neuron model configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            r: Resistance array
        """
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.r = r
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Integrator (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.r.shape}\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        
        s += super().__str__()
        return s

class IF(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple,
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.r = r
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.emits_spike = True
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"IF (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.r.shape}\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_threshold range: [{self.v_threshold.min():.4f}, {self.v_threshold.max():.4f}]\n"
        s += f"\tv_reset range: [{self.v_reset.min():.4f}, {self.v_reset.max():.4f}]\n"
        
        s += super().__str__()
        return s

class LI(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple,
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau = tau
        self.r = r
        self.v_leak = v_leak
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"LI (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tParameter shape: {self.tau.shape}\n"
        s += f"\ttau range: [{self.tau.min():.4f}, {self.tau.max():.4f}]\n"
        s += f"\tr range: [{self.r.min():.4f}, {self.r.max():.4f}]\n"
        s += f"\tv_leak range: [{self.v_leak.min():.4f}, {self.v_leak.max():.4f}]\n"
        
        s += super().__str__()
        return s

class LIF(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple,
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau = tau
        self.r = r
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.emits_spike = True
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"LIF (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
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
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple,
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        
        # Normalize to tuples if they are ints
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"SumPool2d (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        
        s += super().__str__()
        return s

class AvgPool2d(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple,
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
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        
        # Normalize to tuples if they are ints
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"AvgPool2d (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        
        s += super().__str__()
        return s

class Linear(LayerConfig):
    
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, weight: np.ndarray):
        """
        Linear transform without bias configuration.
        
        Args:
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor (calculated by NIR)
            weight: Weight matrix
        """
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.weight = weight
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Linear (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        
        s += super().__str__()
        return s
    
class ModelConfig:
    
    def __init__(self):

        self.layers: List[LayerConfig] = []

        self.input_shape = np.array(0)
        self.output_shape = np.array(0)

        self.merge_count = 0

    def __str__(self):
        
        s = ""
        for idx, layer in enumerate(self.layers):

            if idx > 0:
                s += "\n"

            s += layer.__str__()
        
        return s
    
    def add_layer(self, layer):

        if isinstance(layer, Input):
            self.input_shape = layer.input_shape
            
        elif isinstance(layer, Output):
            self.output_shape = layer.output_shape

        has_created_merge_layer = False

        if len(layer.dependencies) > 1:

            accum_layer_name = get_ready_dependency_layer_name(layer.dependencies)
            has_created_merge_layer = True

            for (dep, is_recurrent) in layer.dependencies:

                if dep == accum_layer_name:
                    continue

                self.merge_count += 1
                merge_layer_name = f"merge_{self.merge_count}"

                merge_layer = Merge(merge_layer_name, accum_layer_name, dep, layer.input_shape)

                merge_layer.add_dependency(accum_layer_name, is_recurrent = False)
                merge_layer.add_dependency(dep, is_recurrent)

                merge_layer.emits_spike = layer_emits_spike(accum_layer_name, self.layers)
                self.layers.append(merge_layer)

        if has_created_merge_layer:
            layer.dependencies.clear()
            layer.add_dependency(accum_layer_name, is_recurrent = False)

        self.layers.append(layer)

def layer_emits_spike(layer_name, layers):

    for layer in layers:

        if layer.name == layer_name:
            return layer.emits_spike
    
    raise Exception(f"Could not find layer '{layer_name}'")

def get_ready_dependency_layer_name(dependencies):

    for (dep_name, is_recurrent) in dependencies:
        if not is_recurrent:
            return dep_name
        
    raise Exception("Unable to find not recurrent layer")
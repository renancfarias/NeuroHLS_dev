from numbers import Real
from typing import List, NamedTuple, Union, Tuple
import warnings
import numpy as np

NUM_DASHES = 55


class TimeDrivenParallelismPlan(NamedTuple):
    """Static reuse architecture selected for one time-driven primitive.

    ``processing_elements`` is the number of arithmetic work items made
    available in a reuse group.  ``reuse_cycles`` is the number of groups
    needed to cover the primitive's static work domain.  The values are
    deliberately recorded together so experiment metadata can distinguish a
    requested percentage from the integer architecture actually generated.
    """

    requested_parallelism: float
    total_work_items: int
    processing_elements: int
    reuse_cycles: int
    effective_parallelism: float
    idle_slots: int
    operation_kind: str

    @property
    def parallelism(self):
        """Compatibility spelling for reports written before this contract."""
        return self.requested_parallelism


def _validate_time_driven_parallelism(parallelism, name="parallelism"):
    if (
        isinstance(parallelism, (bool, np.bool_))
        or not isinstance(parallelism, Real)
    ):
        raise ValueError(
            f"{name} must be a real number in [0, 1], got {parallelism!r}"
        )

    value = float(parallelism)

    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"{name} must be a finite real number in [0, 1], "
            f"got {parallelism!r}"
        )
    return value


def _shape_tuple(shape, name):
    values = tuple(int(value) for value in np.atleast_1d(shape))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain only positive dimensions, got {values}")
    return values


def _positive_tuple(value, dimensions, name):
    if isinstance(value, (int, np.integer)):
        values = (int(value),) * dimensions
    else:
        values = tuple(int(item) for item in np.atleast_1d(value))

    if len(values) != dimensions or any(item <= 0 for item in values):
        raise ValueError(
            f"{name} must have {dimensions} positive value(s), got {values}"
        )
    return values


def _resolve_padding(padding, input_spatial, output_spatial, kernel, stride, dilation):
    dimensions = len(input_spatial)

    if isinstance(padding, str):
        mode = padding.strip().lower()
        if mode == "valid":
            return (0,) * dimensions
        if mode != "same":
            raise ValueError(f"padding must be 'same', 'valid', or integer values, got {padding!r}")

        resolved = []
        for in_size, out_size, kernel_size, step, spacing in zip(
            input_spatial, output_spatial, kernel, stride, dilation
        ):
            effective_kernel = spacing * (kernel_size - 1) + 1
            total_padding = (out_size - 1) * step + effective_kernel - in_size
            if total_padding < 0 or total_padding % 2:
                raise ValueError(
                    "padding='same' requires asymmetric padding for this shape; "
                    "the HLS backend only supports symmetric static padding"
                )
            resolved.append(total_padding // 2)
        return tuple(resolved)

    if isinstance(padding, (int, np.integer)):
        values = (int(padding),) * dimensions
    else:
        values = tuple(int(item) for item in np.atleast_1d(padding))

    if len(values) != dimensions or any(item < 0 for item in values):
        raise ValueError(
            f"padding must have {dimensions} non-negative value(s), got {values}"
        )
    return values


def _validate_convolution_shapes(
    input_shape, output_shape, weight, bias, kernel, stride, padding, dilation, groups
):
    input_shape = _shape_tuple(input_shape, "input_shape")
    output_shape = _shape_tuple(output_shape, "output_shape")
    dimensions = len(kernel)

    if len(input_shape) != dimensions + 1 or len(output_shape) != dimensions + 1:
        raise ValueError(
            f"convolution expects channel plus {dimensions} spatial dimension(s)"
        )
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")

    input_channels = input_shape[0]
    output_channels = output_shape[0]
    if input_channels % groups or output_channels % groups:
        raise ValueError("input and output channels must both be divisible by groups")

    expected_weight_shape = (output_channels, input_channels // groups, *kernel)
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"weight shape must be {expected_weight_shape}, got {tuple(weight.shape)}"
        )
    if bias is not None and tuple(np.asarray(bias).shape) != (output_channels,):
        raise ValueError(
            f"bias shape must be {(output_channels,)}, got {tuple(np.asarray(bias).shape)}"
        )

    calculated_output = []
    for in_size, kernel_size, step, pad, spacing in zip(
        input_shape[1:], kernel, stride, padding, dilation
    ):
        effective_kernel = spacing * (kernel_size - 1) + 1
        calculated_output.append((in_size + 2 * pad - effective_kernel) // step + 1)
    if tuple(calculated_output) != output_shape[1:]:
        raise ValueError(
            f"declared output spatial shape {output_shape[1:]} does not match "
            f"convolution result {tuple(calculated_output)}"
        )

class LayerConfig:

    def __init__(self, name):

        self.is_recurrent = False
        self.dependencies = []
        self.name = name
        self.emits_spike = False
        # ``None`` inherits the model-wide time-driven setting.
        self.time_driven_parallelism = None

    def add_dependency(self, name: str, is_recurrent: bool):
        self.dependencies.append((name, is_recurrent))

    def define_time_driven_parallelism(self, parallelism):
        self.time_driven_parallelism = _validate_time_driven_parallelism(
            parallelism, f"time-driven parallelism for layer {self.name!r}"
        )

    def define_event_driven_parallelism(self, parallelism):
        """Temporary migration shim for the removed event-driven setting.

        The event-driven backend has a scalar stream ABI.  Its previous
        internal actor-lane setting increased hardware cost without a useful
        performance benefit, so a non-zero value is now rejected instead of
        being ignored silently.
        """
        value = _validate_time_driven_parallelism(
            parallelism, f"event-driven parallelism for layer {self.name!r}"
        )
        if value != 0.0:
            raise ValueError(
                "event-driven parallelism was removed; use the scalar "
                "event-driven backend without a p setting"
            )
        warnings.warn(
            "event-driven parallelism is retired and p=0 has no effect",
            DeprecationWarning,
            stacklevel=2,
        )

    def get_neuron_params(self):
        raise Exception(f"{self.name}.get_neuron_params was not implemented")
    
    def get_template_args(self):
        raise Exception(f"{self.name}.get_template_params was not implemented")

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

    def get_neuron_params(self):
        return {}
    
    def get_template_args(self):
        return {}

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

    def get_neuron_params(self):
        return {}
    
    def get_template_args(self):
        return {}

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

    def get_neuron_params(self):
        return {}
    
    def get_template_args(self):
        return {}

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Output ({self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += super().__str__()
        return s
    
class Affine(LayerConfig):
    
    def __init__(self, name: str, weight, bias):

        super().__init__(name)

        self.weight = weight
        self.bias = bias

        self.input_shape = np.atleast_1d(weight.shape[1])
        self.output_shape = np.atleast_1d(weight.shape[0])

    def get_neuron_params(self):
        return {"weights": self.weight,
                "bias": self.bias}
    
    def get_template_args(self):
        # Parallelism is resolved centrally from the time-driven `p` contract.
        return {}

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

    def get_neuron_params(self):
        return {}
    
    def get_template_args(self):
        return {}
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Flatten (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        # s += f"\tstart_dim={self.start_dim}, end_dim={self.end_dim}\n"
        s += "-" * NUM_DASHES + "\n"
        
        s += super().__str__()
        return s

class Conv1d(LayerConfig):
    
    def __init__(self, name: str,
                 input_shape: tuple,
                 output_shape: tuple,
                 weight: np.ndarray, 
                 stride: int,
                 padding: Union[int, str],
                 dilation: int, 
                 groups: int,
                 bias: np.ndarray):
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
        self.input_shape = np.asarray(input_shape)
        self.output_shape = np.asarray(output_shape)

        self.weight = np.asarray(weight)
        self.stride = _positive_tuple(stride, 1, "stride")
        self.dilation = _positive_tuple(dilation, 1, "dilation")
        self.kernel = (int(self.weight.shape[2]),)
        self.padding = _resolve_padding(
            padding,
            _shape_tuple(self.input_shape, "input_shape")[1:],
            _shape_tuple(self.output_shape, "output_shape")[1:],
            self.kernel,
            self.stride,
            self.dilation,
        )
        self.groups = int(groups)
        self.bias = None if bias is None else np.asarray(bias)

        _validate_convolution_shapes(
            self.input_shape, self.output_shape, self.weight, self.bias,
            self.kernel, self.stride, self.padding, self.dilation, self.groups
        )

    def get_neuron_params(self):
        params = {"weights": self.weight}
        if self.bias is not None:
            params["bias"] = self.bias
        return params
    
    def get_template_args(self):
        return {"kernel": self.kernel,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups}
    
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
    
    def __init__(self, 
                 name: str,
                 input_shape: tuple,
                 output_shape: tuple,
                 weight: np.ndarray,
                 stride: Union[int, Tuple[int, int]], 
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]], 
                 groups: int,
                 bias: np.ndarray):
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
        self.input_shape = np.asarray(input_shape)
        self.output_shape = np.asarray(output_shape)
        self.weight = np.asarray(weight)
        
        # Normalize stride, padding, dilation to tuples if they are ints
        self.stride = _positive_tuple(stride, 2, "stride")
        self.dilation = _positive_tuple(dilation, 2, "dilation")
        
        self.groups = int(groups)
        self.bias = None if bias is None else np.asarray(bias)

        self.kernel = (self.weight.shape[2], self.weight.shape[3])
        self.padding = _resolve_padding(
            padding,
            _shape_tuple(self.input_shape, "input_shape")[1:],
            _shape_tuple(self.output_shape, "output_shape")[1:],
            self.kernel,
            self.stride,
            self.dilation,
        )

        _validate_convolution_shapes(
            self.input_shape, self.output_shape, self.weight, self.bias,
            self.kernel, self.stride, self.padding, self.dilation, self.groups
        )

    def get_neuron_params(self):
        params = {"weights": self.weight}
        if self.bias is not None:
            params["bias"] = self.bias
        return params
    
    def get_template_args(self):
        return {"kernel": self.kernel,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups}
    
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

    def get_neuron_params(self):
        return {"tau_syn": self.tau_syn,
                "tau_mem": self.tau_mem,
                "r": self.r,
                "v_leak": self.v_leak,
                "w_in": self.w_in}
    
    def get_template_args(self):
        return {}
    
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
                 v_threshold: np.ndarray, v_reset: np.ndarray, w_in: np.ndarray,
                 reset_by_subtraction: bool = False):
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
        self.dt = 0.0001
        self.reset_by_subtraction = reset_by_subtraction
        self.emits_spike = True

    def get_neuron_params(self):
        return {"tau_syn": self.tau_syn,
                "tau_mem": self.tau_mem,
                "r": self.r,
                "v_leak": self.v_leak,
                "v_threshold": self.v_threshold,
                "v_reset": self.v_reset,
                "w_in": self.w_in,
                "u_state": np.zeros(self.output_shape),
                "v_state": np.zeros(self.output_shape)}   # VER COMO ADD PARAMETRO 'dt'
    
    def get_template_args(self):
        return {}
    
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

    def get_neuron_params(self):
        return {"r": self.r}
    
    def get_template_args(self):
        return {}
    
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

    def get_neuron_params(self):
        return {"r": self.r,
                "v_threshold": self.v_threshold,
                "v_reset": self.v_reset}
    
    def get_template_args(self):
        return {}
    
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

    def get_neuron_params(self):
        return {"tau": self.tau,
                "r": self.r,
                "v_leak": self.v_leak}
    
    def get_template_args(self):
        return {}
    
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
                 v_threshold: np.ndarray, v_reset: np.ndarray,
                 reset_by_subtraction: bool = False):
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
            reset_by_subtraction: Subtract v_threshold on spike instead of
                assigning v_reset, matching the NIR reference semantics
        """
        super().__init__(name)
        self.input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
        self.output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
        self.tau = tau
        self.r = r
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.reset_by_subtraction = reset_by_subtraction
        self.emits_spike = True

    def get_neuron_params(self):
        return {"tau": self.tau,
                "r": self.r,
                "v_leak": self.v_leak,
                "v_threshold": self.v_threshold,
                "v_reset": self.v_reset}
    
    def get_template_args(self):
        return {}
    
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

    def get_neuron_params(self):
        return {}
    
    def get_template_args(self):
        return {"kernel": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding}
    
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
        self.input_shape = np.asarray(input_shape)
        self.output_shape = np.asarray(output_shape)

        if self.input_shape.ndim != 1 or self.output_shape.ndim != 1 or len(self.input_shape) != 3 or len(self.output_shape) != 3:
            raise ValueError("AvgPool2d expects shapes (channels, height, width)")
        
        # Normalize to tuples if they are ints
        self.kernel_size = _positive_tuple(kernel_size, 2, "kernel_size")
        self.stride = _positive_tuple(stride, 2, "stride")
        self.padding = _resolve_padding(
            padding,
            _shape_tuple(self.input_shape, "input_shape")[1:],
            _shape_tuple(self.output_shape, "output_shape")[1:],
            self.kernel_size,
            self.stride,
            (1, 1),
        )

        calculated_output = tuple(
            (int(size) + 2 * pad - kernel) // step + 1
            for size, pad, kernel, step in zip(
                self.input_shape[1:], self.padding, self.kernel_size, self.stride
            )
        )
        if self.input_shape[0] != self.output_shape[0] or calculated_output != tuple(self.output_shape[1:]):
            raise ValueError(
                f"AvgPool2d output shape {tuple(self.output_shape)} does not match "
                f"expected {(int(self.input_shape[0]), *calculated_output)}"
            )

    def get_neuron_params(self):
        return {}
    
    def get_template_args(self):
        return {"kernel": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding}
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"AvgPool2d (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tKernel size: {self.kernel_size}\n"
        s += f"\tStride: {self.stride}\n"
        s += f"\tPadding: {self.padding}\n"
        
        s += super().__str__()
        return s


class Scale(LayerConfig):

    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, scale: np.ndarray):
        super().__init__(name)
        self.input_shape = np.asarray(input_shape)
        self.output_shape = np.asarray(output_shape)
        raw_scale = np.asarray(scale)

        input_dims = _shape_tuple(self.input_shape, "input_shape")
        output_dims = _shape_tuple(self.output_shape, "output_shape")
        if input_dims != output_dims:
            raise ValueError(
                f"Scale must preserve shape, got input {input_dims} and output {output_dims}"
            )
        if len(input_dims) not in (1, 2, 3):
            raise ValueError("Scale supports only 1D, 2D, or 3D tensors")
        if raw_scale.size != 1 and tuple(raw_scale.shape) != input_dims:
            raise ValueError(
                f"Scale parameter must be scalar or have shape {input_dims}, "
                f"got {tuple(raw_scale.shape)}"
            )

        self.scalar_broadcast = raw_scale.size == 1
        self.scale = (
            np.full(input_dims, raw_scale.reshape(-1)[0], dtype=raw_scale.dtype)
            if self.scalar_broadcast
            else raw_scale
        )

    def get_neuron_params(self):
        return {"scale": self.scale}

    def get_template_args(self):
        return {}

    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Scale (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"
        s += f"\tScale shape: {self.scale.shape}\n"
        s += f"\tScalar broadcast: {'YES' if self.scalar_broadcast else 'NO'}\n"
        return s + super().__str__()

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
    def get_neuron_params(self):
        return {"weights": self.weight}
    
    def get_template_args(self):
        # Parallelism is resolved centrally from the time-driven `p` contract.
        return {}
    
    def __str__(self):
        s = "-" * NUM_DASHES + "\n"
        s += f"Linear (input: {self.input_shape}, output: {self.output_shape}) - layer name: '{self.name}'\n"
        s += "-" * NUM_DASHES + "\n"

        s += f"\tWeight shape: {self.weight.shape}\n"
        
        s += super().__str__()
        return s


def _time_driven_work_domain(layer):
    """Return the number and category of statically scheduled operations.

    The count deliberately includes padded kernel positions.  Those slots may
    be guarded at run time, but they are part of the fixed loop nest and hence
    of the architecture selected by ``p``.
    """
    supported_types = (
        Linear, Affine, Conv1d, Conv2d, SumPool2d, AvgPool2d,
        Merge, Flatten, Scale, IF, LIF, CubaLIF,
    )
    if not isinstance(layer, supported_types):
        raise ValueError(
            f"time-driven parallelism is not supported for "
            f"{type(layer).__name__} layer {layer.name!r}"
        )

    input_shape = _shape_tuple(layer.input_shape, "input_shape")
    output_shape = _shape_tuple(layer.output_shape, "output_shape")
    if isinstance(layer, (Merge, Flatten, Scale, IF, LIF, CubaLIF)):
        if len(input_shape) not in (1, 2, 3):
            raise ValueError(
                f"time-driven {type(layer).__name__} supports only 1D, "
                f"2D, or 3D tensors, got rank {len(input_shape)}"
            )

    if isinstance(layer, Flatten):
        expected_output = (int(np.prod(input_shape)),)
        if output_shape != expected_output:
            raise ValueError(
                f"time-driven Flatten expects output shape "
                f"{expected_output}, got {output_shape}"
            )

    output_elements = int(np.prod(output_shape))
    if isinstance(layer, (Linear, Affine)):
        return output_elements * int(np.prod(input_shape)), "mac"
    if isinstance(layer, Conv1d):
        terms = (int(layer.input_shape[0]) // int(layer.groups)) * int(layer.kernel[0])
        return output_elements * terms, "conv1d_mac"
    if isinstance(layer, Conv2d):
        terms = (
            (int(layer.input_shape[0]) // int(layer.groups))
            * int(layer.kernel[0])
            * int(layer.kernel[1])
        )
        return output_elements * terms, "conv2d_mac"
    if isinstance(layer, (SumPool2d, AvgPool2d)):
        terms = int(layer.kernel_size[0]) * int(layer.kernel_size[1])
        return output_elements * terms, "pool_accumulate"
    if isinstance(layer, (IF, LIF, CubaLIF)):
        return output_elements, "neuron_update"
    return output_elements, "elementwise"


def resolve_time_driven_parallelism_plan(layer, parallelism):
    """Resolve normalized ``p`` to a static percent-parallel reuse plan.

    ``p=0`` is the explicit serial sentinel.  For positive values, rounding is
    half-up so that the generated architecture is stable across Python versions
    and does not inherit bankers' rounding from :func:`round`.
    """
    value = _validate_time_driven_parallelism(parallelism)
    work_items, operation_kind = _time_driven_work_domain(layer)
    if work_items <= 0:
        raise ValueError(
            f"time-driven work domain for layer {layer.name!r} must be positive"
        )

    if value == 0.0:
        processing_elements = 1
    else:
        processing_elements = min(
            work_items, max(1, int(np.floor(value * work_items + 0.5)))
        )
    reuse_cycles = (work_items + processing_elements - 1) // processing_elements
    idle_slots = reuse_cycles * processing_elements - work_items

    return TimeDrivenParallelismPlan(
        requested_parallelism=value,
        total_work_items=work_items,
        processing_elements=processing_elements,
        reuse_cycles=reuse_cycles,
        effective_parallelism=processing_elements / work_items,
        idle_slots=idle_slots,
        operation_kind=operation_kind,
    )


class ModelConfig:
    
    def __init__(self):

        self.layers: List[LayerConfig] = []

        self.input_shape = np.array(0)
        self.output_shape = np.array(0)

        self.merge_count = 0

        self.input_quantization = (16, 8)
        self.weight_quantization = (16, 8)
        self.potential_quantization = (16, 8)

        # Architecture controls for the time-driven backend.  A layer-level
        # value of ``None`` inherits this model-wide setting.
        self.time_driven_parallelism = 0.0

        # Physical duration (seconds) of an event-driven input step.  When
        # unset, the generator infers it from CubaLIF layers and otherwise
        # falls back to the project convention of 100 us.
        self.event_dt = None
        # PWL is the sole supported event-driven decay approximation.
        self.event_decay_approximation = "piecewise_linear"
        # ``discrete_compatible`` treats one input bin as a finite Euler
        # step: the generator converts w_in into the effective current jump
        # (dt / tau_syn) * w_in and keeps at most one spike per logical step.
        self.event_cuba_lif_mode = "discrete_compatible"
        # Active-list execution updates only active neurons on the
        # END_STEP/END_SAMPLE lightweight tick.
        self.event_cuba_lif_strategy = "active_list"
        self.event_active_noise_threshold = 1e-6

        # Original NIR graph, retained for graph-level backend rewrites.
        self.graph_edges = []
        self.graph_layers = {}

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

    def define_input_quantization(self, total_bits, int_bits):
        self.input_quantization = (total_bits, int_bits)

    def define_weight_quantization(self, total_bits, int_bits):
        self.weight_quantization = (total_bits, int_bits)

    def define_potential_quantization(self, total_bits, int_bits):
        self.potential_quantization = (total_bits, int_bits)

    def define_time_driven_parallelism(self, parallelism):
        self.time_driven_parallelism = _validate_time_driven_parallelism(
            parallelism, "time-driven parallelism"
        )

    def define_time_driven_layer_parallelism(self, layer_name, parallelism):
        self._find_layer_by_name(layer_name).define_time_driven_parallelism(
            parallelism
        )

    def define_event_driven_parallelism(self, parallelism):
        """Migration shim for code written before event lanes were retired."""
        value = _validate_time_driven_parallelism(
            parallelism, "event-driven parallelism"
        )
        if value != 0.0:
            raise ValueError(
                "event-driven parallelism was removed; configure p only for "
                "the time-driven backend"
            )
        warnings.warn(
            "event-driven parallelism is retired and p=0 has no effect",
            DeprecationWarning,
            stacklevel=2,
        )

    def define_layer_parallelism(self, layer_name, parallelism):
        warnings.warn(
            "define_layer_parallelism is deprecated; use "
            "define_time_driven_layer_parallelism",
            DeprecationWarning,
            stacklevel=2,
        )
        self.define_time_driven_layer_parallelism(layer_name, parallelism)

    def _find_layer_by_name(self, layer_name):
        candidates = list(self.layers)
        for layer in (self.graph_layers or {}).values():
            if all(layer is not existing for existing in candidates):
                candidates.append(layer)
        matches = [layer for layer in candidates if layer.name == layer_name]
        if not matches:
            raise ValueError(f"could not find layer {layer_name!r}")
        if len(matches) > 1:
            raise ValueError(f"layer name {layer_name!r} is ambiguous")
        return matches[0]

    def define_event_layer_parallelism(self, layer_name, parallelism):
        """Migration shim that rejects removed non-scalar event lanes."""
        value = _validate_time_driven_parallelism(
            parallelism, "event-driven parallelism"
        )
        if value != 0.0:
            raise ValueError(
                "event-driven parallelism was removed; configure p only for "
                "the time-driven backend"
            )
        warnings.warn(
            "event-driven parallelism is retired and p=0 has no effect",
            DeprecationWarning,
            stacklevel=2,
        )

    def resolve_time_driven_parallelism(self, layer):
        if isinstance(layer, str):
            layer = self._find_layer_by_name(layer)

        override = getattr(layer, "time_driven_parallelism", None)
        parallelism = (
            self.time_driven_parallelism if override is None else override
        )
        return resolve_time_driven_parallelism_plan(
            layer,
            parallelism,
        )

    def define_event_dt(self, dt):
        event_dt = float(dt)
        if not np.isfinite(event_dt) or event_dt <= 0:
            raise ValueError(f"event_dt must be finite and greater than zero, got {dt!r}")
        self.event_dt = event_dt

    def define_event_decay_approximation(self, approximation):
        normalized = str(approximation).strip().lower().replace("-", "_")
        aliases = {
            "pwl": "piecewise_linear",
            "piecewise": "piecewise_linear",
            "piecewise_linear": "piecewise_linear",
        }
        if normalized not in aliases:
            raise ValueError(
                "only the 'piecewise_linear' event decay approximation is "
                f"supported; got {approximation!r}"
            )
        self.event_decay_approximation = aliases[normalized]

    def define_event_cuba_lif_mode(self, mode):
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "discrete": "discrete_compatible",
            "step": "discrete_compatible",
            "discrete_compatible": "discrete_compatible",
        }
        if normalized not in aliases:
            raise ValueError(
                "only the 'discrete_compatible' event CubaLIF mode is "
                f"supported; got {mode!r}"
            )
        self.event_cuba_lif_mode = aliases[normalized]

    def define_event_cuba_lif_strategy(self, strategy):
        normalized = str(strategy).strip().lower().replace("-", "_")
        aliases = {
            "active": "active_list",
            "active_list": "active_list",
            "hybrid": "active_list",
            "lightweight_ticks": "active_list",
        }
        if normalized not in aliases:
            raise ValueError(
                "only the 'active_list' event CubaLIF strategy is "
                f"supported; got {strategy!r}"
            )
        self.event_cuba_lif_strategy = aliases[normalized]

    def define_event_active_noise_threshold(self, threshold):
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(
                "event active-list noise threshold must be finite and "
                f"greater than zero, got {threshold!r}"
            )
        self.event_active_noise_threshold = threshold

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

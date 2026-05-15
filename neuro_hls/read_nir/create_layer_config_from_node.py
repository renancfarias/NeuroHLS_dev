from .layer_configuration import *

def _metadata_bool(value):

    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")

    return bool(value)

def create_layer_config_from_node(node_name, node):
    """
    Extrai informações de um nó NIR e retorna a instância da classe apropriada.
    
    Args:
        node: Nó do grafo NIR
        input_shape: Shape da entrada para esta camada
        output_shape: Shape da saída desta camada
        
    Returns:
        Instância de LayerConfig apropriada ou None se o tipo não for reconhecido
    """

    input_shape = node.input_type['input']
    output_shape = node.output_type['output']
    node_type = type(node).__name__
    
    # Affine layer (camada linear com bias)
    if node_type == 'Affine':
        # Affine tem weight de shape [out, in] e bias de shape [out]
        if hasattr(node, 'weight') and node.weight is not None:
            return Affine(node_name, node.weight, node.bias)
        else:
            raise ValueError(f"Nó Affine sem weight: {node}")
    
    # Flatten layer
    elif node_type == 'Flatten':
        start_dim = getattr(node, 'start_dim', 1)
        end_dim = getattr(node, 'end_dim', -1)
        return Flatten(node_name, input_shape, output_shape, start_dim, end_dim)
    
    # Conv1d layer
    elif node_type == 'Conv1d':
        weight = node.weight
        stride = getattr(node, 'stride', 1)
        padding = getattr(node, 'padding', 0)
        dilation = getattr(node, 'dilation', 1)
        groups = getattr(node, 'groups', 1)
        bias = getattr(node, 'bias', None)
        
        return Conv1d(node_name, input_shape, output_shape, weight, stride, padding, dilation, groups, bias)
    
    # Conv2d layer
    elif node_type == 'Conv2d':
        weight = node.weight
        stride = getattr(node, 'stride', 1)
        padding = getattr(node, 'padding', 0)
        dilation = getattr(node, 'dilation', 1)
        groups = getattr(node, 'groups', 1)
        bias = getattr(node, 'bias', None)
        
        return Conv2d(node_name, input_shape, output_shape, weight, stride, padding, dilation, groups, bias)
    
    # CubaLI (Current-based Leaky Integrator)
    elif node_type == 'CubaLI':
        tau_syn = node.tau_syn
        tau_mem = node.tau_mem
        r = node.r
        v_leak = node.v_leak
        w_in = node.w_in
        
        return CubaLI(node_name, input_shape, output_shape, tau_syn, tau_mem, r, v_leak, w_in)
    
    # CubaLIF (Current-based Leaky Integrate-and-Fire)
    elif node_type == 'CubaLIF':
        tau_syn = node.tau_syn
        tau_mem = node.tau_mem
        r = node.r
        v_leak = node.v_leak
        v_threshold = node.v_threshold
        v_reset = getattr(node, "v_reset", np.zeros_like(v_threshold))
        w_in = node.w_in
        metadata = getattr(node, "metadata", {}) or {}
        reset_mechanism = metadata.get("reset_mechanism", getattr(node, "reset_mechanism", None))
        reset_mechanism = reset_mechanism.strip().lower() if isinstance(reset_mechanism, str) else reset_mechanism
        reset_by_subtraction = reset_mechanism == "subtract" or _metadata_bool(metadata.get("reset_by_subtraction", False))
        
        return CubaLIF(node_name, input_shape, output_shape, tau_syn, tau_mem, r, v_leak, v_threshold, v_reset, w_in, reset_by_subtraction)
    
    # I (Integrator)
    elif node_type == 'I':
        r = node.r
        return I(node_name, input_shape, output_shape, r)
    
    # IF (Integrate-and-Fire)
    elif node_type == 'IF':
        r = node.r
        v_threshold = node.v_threshold
        v_reset = getattr(node, "v_reset", np.zeros_like(v_threshold))
        
        return IF(node_name, input_shape, output_shape, r, v_threshold, v_reset)
    
    # LI (Leaky Integrator)
    elif node_type == 'LI':
        tau = node.tau
        r = node.r
        v_leak = node.v_leak
        
        return LI(node_name, input_shape, output_shape, tau, r, v_leak)
    
    # LIF (Leaky Integrate-and-Fire)
    elif node_type == 'LIF':
        tau = node.tau
        r = node.r
        v_leak = node.v_leak
        v_threshold = node.v_threshold
        v_reset = getattr(node, "v_reset", np.zeros_like(v_threshold))
        
        return LIF(node_name, input_shape, output_shape, tau, r, v_leak, v_threshold, v_reset)
    
    # SumPool2d
    elif node_type == 'SumPool2d':
        kernel_size = getattr(node, 'kernel_size', 2)
        stride = getattr(node, 'stride', None)
        if stride is None:
            stride = kernel_size
        padding = getattr(node, 'padding', 0)
        
        return SumPool2d(node_name, input_shape, output_shape, kernel_size, stride, padding)
    
    # AvgPool2d
    elif node_type == 'AvgPool2d':
        kernel_size = getattr(node, 'kernel_size', 2)
        stride = getattr(node, 'stride', None)
        if stride is None:
            stride = kernel_size
        padding = getattr(node, 'padding', 0)
        
        return AvgPool2d(node_name, input_shape, output_shape, kernel_size, stride, padding)
    
    # Linear (sem bias)
    elif node_type == 'Linear':
        weight = node.weight
        return Linear(node_name, input_shape, output_shape, weight)
    
    elif node_type == "Input":
        return Input(node_name, input_shape)
    
    elif node_type == "Output":
        return Output(node_name, output_shape)
    
    # Tipo não reconhecido
    else:
        raise ValueError(f"Tipo de nó não reconhecido: {node_type}")

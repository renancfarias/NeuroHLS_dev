"""
Debug script para investigar carregamento de pesos do arquivo .nir
Verifica se os pesos estão sendo lidos corretamente e se a ordem está preservada
"""

import nir
import numpy as np
import sys
from pathlib import Path

# Adicionar caminho do módulo neuro_hls
sys.path.append(str(Path(__file__).resolve().parent))

from neuro_hls.read_nir import get_model_config_from_nir

def print_separator(title=""):
    if title:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    else:
        print("-" * 70)

def inspect_nir_file(nir_file_path):
    """
    Inspeciona um arquivo .nir e imprime informações detalhadas sobre:
    - Estrutura da rede
    - Pesos de cada camada
    - Tipos de nós
    """
    
    print_separator(f"Investigando: {nir_file_path}")
    
    # Carregar graph NIR diretamente
    print("\n[1] Carregando arquivo .nir diretamente com nir.read()...")
    nir_graph = nir.read(nir_file_path)
    
    nodes = nir_graph.nodes
    edges = nir_graph.edges
    
    print(f"  Total de nós: {len(nodes)}")
    print(f"  Total de arestas: {len(edges)}")
    
    # Imprimir estrutura
    print_separator("[2] Estrutura da Rede (nós e conexões)")
    
    print("\nNós encontrados:")
    for i, (node_name, node) in enumerate(nodes.items(), 1):
        node_type = type(node).__name__
        print(f"  {i}. '{node_name}' -> {node_type}")
    
    print("\nConexões (arestas):")
    for i, edge in enumerate(edges, 1):
        print(f"  {i}. {edge[0]} -> {edge[1]}")
    
    # Inspecionar pesos de cada camada
    print_separator("[3] Pesos de Cada Camada")
    
    for node_name, node in nodes.items():
        node_type = type(node).__name__
        
        print(f"\n--- {node_name} ({node_type}) ---")
        
        # Affine / Linear
        if node_type == 'Affine':
            if hasattr(node, 'weight') and node.weight is not None:
                print(f"  Weight shape: {node.weight.shape}")
                print(f"  Weight dtype: {node.weight.dtype}")
                print(f"  Weight min: {node.weight.min():.6f}, max: {node.weight.max():.6f}")
                print(f"  Weight primeiras 3 linhas (primeiros 5 valores):")
                for i in range(min(3, node.weight.shape[0])):
                    print(f"    [{i}]: {node.weight[i, :min(5, node.weight.shape[1])]}")
                
                if hasattr(node, 'bias') and node.bias is not None:
                    print(f"  Bias shape: {node.bias.shape}")
                    print(f"  Bias: {node.bias[:min(10, len(node.bias))]}")
                else:
                    print(f"  Bias: None")
            else:
                print("  ERRO: Affine sem weight!")
        
        # Conv2d
        elif node_type == 'Conv2d':
            if hasattr(node, 'weight') and node.weight is not None:
                print(f"  Weight shape: {node.weight.shape}")
                print(f"  Weight dtype: {node.weight.dtype}")
                print(f"  Weight min: {node.weight.min():.6f}, max: {node.weight.max():.6f}")
                
                if hasattr(node, 'stride'):
                    print(f"  Stride: {node.stride}")
                if hasattr(node, 'padding'):
                    print(f"  Padding: {node.padding}")
                if hasattr(node, 'dilation'):
                    print(f"  Dilation: {node.dilation}")
                if hasattr(node, 'groups'):
                    print(f"  Groups: {node.groups}")
                
                if hasattr(node, 'bias') and node.bias is not None:
                    print(f"  Bias shape: {node.bias.shape}")
                    print(f"  Bias: {node.bias[:min(10, len(node.bias))]}")
                else:
                    print(f"  Bias: None")
        
        # CubaLIF
        elif node_type == 'CubaLIF':
            print(f"  Parâmetros do neurônio:")
            for param_name in ['tau_syn', 'tau_mem', 'r', 'v_leak', 'v_threshold', 'w_in']:
                if hasattr(node, param_name):
                    param = getattr(node, param_name)
                    if param is not None:
                        if isinstance(param, (int, float)):
                            print(f"    {param_name}: {param}")
                        else:
                            print(f"    {param_name} shape: {param.shape if hasattr(param, 'shape') else 'scalar'}")
                            if hasattr(param, '__len__'):
                                print(f"      valores: {param[:min(5, len(param))]}")
                            else:
                                print(f"      valor: {param}")
        
        # LIF
        elif node_type == 'LIF':
            print(f"  Parâmetros do neurônio:")
            for param_name in ['tau', 'r', 'v_leak', 'v_threshold']:
                if hasattr(node, param_name):
                    param = getattr(node, param_name)
                    if param is not None:
                        print(f"    {param_name}: {param if isinstance(param, (int, float)) else f'array shape {param.shape}'}")
        
        # SumPool2d
        elif node_type == 'SumPool2d':
            print(f"  Kernel size: {node.kernel_size if hasattr(node, 'kernel_size') else 'N/A'}")
            print(f"  Stride: {node.stride if hasattr(node, 'stride') else 'N/A'}")
            print(f"  Padding: {node.padding if hasattr(node, 'padding') else 'N/A'}")
        
        # Flatten
        elif node_type == 'Flatten':
            print(f"  start_dim: {node.start_dim if hasattr(node, 'start_dim') else 1}")
            print(f"  end_dim: {node.end_dim if hasattr(node, 'end_dim') else -1}")
        
        # Input/Output
        elif node_type in ['Input', 'Output']:
            if hasattr(node, 'input_type'):
                print(f"  input_type: {node.input_type}")
            if hasattr(node, 'output_type'):
                print(f"  output_type: {node.output_type}")
        
        else:
            print(f"  Tipo desconhecido: {node_type}")
            print(f"  Atributos: {dir(node)}")
    
    # Agora testar o carregamento via get_model_config_from_nir
    print_separator("[4] Carregamento via get_model_config_from_nir()")
    
    try:
        model_config = get_model_config_from_nir(nir_file_path)
        
        print(f"\nInput shape: {model_config.input_shape}")
        print(f"Output shape: {model_config.output_shape}")
        
        # model_config.layers é uma lista, não dicionário
        if isinstance(model_config.layers, dict):
            layers_list = list(model_config.layers.items())
        else:
            layers_list = [(layer.layer_name if hasattr(layer, 'layer_name') else f'layer_{i}', layer) 
                          for i, layer in enumerate(model_config.layers)]
        
        print(f"Total de camadas: {len(layers_list)}")
        
        print("\nCamadas processadas:")
        for i, (layer_name, layer) in enumerate(layers_list, 1):
            layer_type = type(layer).__name__
            print(f"  {i}. '{layer_name}' -> {layer_type}")
            
            # Verificar se os pesos foram carregados
            if hasattr(layer, 'weights'):
                print(f"      Weights shape: {layer.weights.shape}")
                print(f"      Weights primeiros valores: {layer.weights.flatten()[:5]}")
            
            if hasattr(layer, 'bias'):
                if layer.bias is not None:
                    print(f"      Bias shape: {layer.bias.shape if hasattr(layer.bias, 'shape') else 'scalar'}")
                    print(f"      Bias primeiros valores: {layer.bias.flatten()[:5] if hasattr(layer.bias, 'flatten') else layer.bias}")
            
            # CubaLIF parâmetros
            if layer_type == 'CubaLIF':
                print(f"      Parâmetros do neurônio:")
                for attr in ['tau_syn', 'tau_mem', 'R', 'v_leak', 'v_threshold', 'v_reset', 'w_in', 'dt']:
                    if hasattr(layer, attr):
                        val = getattr(layer, attr)
                        if val is not None:
                            if hasattr(val, 'shape'):
                                print(f"        {attr} (shape {val.shape}): {val[:min(3, len(val))]}")
                            else:
                                print(f"        {attr}: {val}")
    
    except Exception as e:
        print(f"\n  ERRO ao carregar com get_model_config_from_nir(): {e}")
        import traceback
        traceback.print_exc()
    
    print_separator()

if __name__ == "__main__":
    
    # Procurar arquivos .nir na raiz do projeto
    nir_files = list(Path(".").glob("*.nir"))
    
    if len(nir_files) == 0:
        print("Nenhum arquivo .nir encontrado no diretório atual")
        sys.exit(1)
    
    print("Arquivos .nir encontrados:")
    for i, nir_file in enumerate(nir_files, 1):
        print(f"  {i}. {nir_file}")
    
    # Se especificado na linha de comando, usar esse arquivo
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Usar o primeiro arquivo encontrado
        target_file = str(nir_files[0])
    
    if not Path(target_file).exists():
        print(f"\nERRO: Arquivo '{target_file}' não encontrado!")
        sys.exit(1)
    
    inspect_nir_file(target_file)

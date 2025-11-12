#!/usr/bin/env python3
"""
NIR to C++ Parser
Converte um gráfico NIR em chamadas de funções C++ vazias correspondentes às primitivas da rede.
Também extrai os pesos do arquivo NIR e gera um arquivo .h com os pesos organizados.
"""

import nir
import numpy as np
import os
import re
from typing import Dict, List, Tuple, Any, Iterable


class NIRToCppParser:
    """Parser que converte um gráfico NIR em funções C++ vazias."""
    
    def __init__(self, nir_file: str):
        """
        Inicializa o parser com um arquivo NIR.
        
        Args:
            nir_file: Caminho para o arquivo .nir
        """
        self.nir_graph = nir.read(nir_file)
        self.nodes = self.nir_graph.nodes
        self.edges = self.nir_graph.edges
        self.execution_order = []
        self._build_execution_order()
    
    def _build_execution_order(self):
        """Constrói a ordem de execução baseada nas arestas do gráfico."""
        # Cria um mapa de dependências
        dependencies = {name: [] for name in self.nodes.keys()}
        
        # Popula as dependências baseado nas arestas
        for src, dst in self.edges:
            if dst in dependencies:
                dependencies[dst].append(src)
        
        # Ordena topologicamente
        visited = set()
        temp_visited = set()
        
        def visit(node_name):
            if node_name in temp_visited:
                raise ValueError(f"Dependência circular detectada em {node_name}")
            if node_name in visited:
                return
            
            temp_visited.add(node_name)
            for dep in dependencies.get(node_name, []):
                visit(dep)
            temp_visited.remove(node_name)
            visited.add(node_name)
            
            # Adiciona apenas nós que não são input/output
            if node_name not in ['input', 'output']:
                self.execution_order.append(node_name)
        
        # Visita todos os nós
        for node_name in self.nodes.keys():
            if node_name not in visited:
                visit(node_name)
    
    def _get_node_params(self, node: Any) -> Dict[str, Any]:
        """Extrai parâmetros relevantes de um nó NIR."""
        params = {}
        
        if hasattr(node, 'weight') and node.weight is not None:
            params['weight_shape'] = list(node.weight.shape)
            
        if hasattr(node, 'bias') and node.bias is not None:
            params['bias_shape'] = list(node.bias.shape)
            
        if hasattr(node, 'kernel_size'):
            params['kernel_size'] = int(node.kernel_size) if np.isscalar(node.kernel_size) else list(node.kernel_size)
            
        if hasattr(node, 'stride'):
            params['stride'] = int(node.stride) if np.isscalar(node.stride) else list(node.stride)
            
        if hasattr(node, 'padding'):
            params['padding'] = list(node.padding) if hasattr(node.padding, '__iter__') else [node.padding]
            
        if hasattr(node, 'tau') and node.tau is not None:
            params['tau_shape'] = list(node.tau.shape)
            
        if hasattr(node, 'r') and node.r is not None:
            params['r_shape'] = list(node.r.shape)
            
        if hasattr(node, 'v_threshold') and node.v_threshold is not None:
            params['v_threshold_shape'] = list(node.v_threshold.shape)
            
        if hasattr(node, 'v_leak') and node.v_leak is not None:
            params['v_leak_shape'] = list(node.v_leak.shape)
            
        if hasattr(node, 'v_reset') and node.v_reset is not None:
            params['v_reset_shape'] = list(node.v_reset.shape)
            
        if hasattr(node, 'input_shape') and node.input_shape is not None:
            params['input_shape'] = list(node.input_shape)
            
        if hasattr(node, 'start_dim'):
            params['start_dim'] = int(node.start_dim)
            
        if hasattr(node, 'end_dim'):
            params['end_dim'] = int(node.end_dim)
            
        return params
    
    def _generate_function_call(self, node_name: str, node: Any) -> str:
        """Gera uma chamada de função C++ para um nó NIR."""
        node_type = type(node).__name__
        params = self._get_node_params(node)
        
        # Mapeia tipos NIR para nomes de funções C++
        function_mapping = {
            'Conv2d': 'conv2d_layer',
            'AvgPool2d': 'avgpool2d_layer', 
            'LIF': 'lif_neuron_layer',
            'Affine': 'linear_layer',
            'Flatten': 'flatten_layer'
        }
        
        function_name = function_mapping.get(node_type, f"unknown_layer_{node_type.lower()}")
        
        # Gera comentário com parâmetros
        comment_lines = [f"    // Node {node_name}: {node_type}"]
        for param_name, param_value in params.items():
            comment_lines.append(f"    // {param_name}: {param_value}")
        
        # Gera chamada da função
        call_line = f"    {function_name}(); // Layer {node_name}"
        
        return "\n".join(comment_lines + [call_line])
    
    def generate_cpp_header(self) -> str:
        """Gera o header C++ com declarações das funções."""
        header = [
            "#ifndef NIR_NETWORK_H",
            "#define NIR_NETWORK_H",
            "",
            "// Funções das primitivas da rede neural (implementações vazias por enquanto)",
            "",
            "// Camadas de convolução",
            "void conv2d_layer();",
            "",
            "// Camadas de pooling", 
            "void avgpool2d_layer();",
            "",
            "// Neurônios LIF (Leaky Integrate-and-Fire)",
            "void lif_neuron_layer();",
            "",
            "// Camadas lineares/totalmente conectadas",
            "void linear_layer();",
            "",
            "// Camadas de flatten",
            "void flatten_layer();",
            "",
            "// Função principal que executa a rede",
            "void execute_network();",
            "",
            "#endif // NIR_NETWORK_H"
        ]
        return "\n".join(header)
    
    def generate_cpp_implementation(self) -> str:
        """Gera a implementação C++ com as chamadas das funções."""
        implementation = [
            '#include "nir_network.h"',
            '#include <iostream>',
            "",
            "// Implementações vazias das primitivas da rede",
            "",
            "void conv2d_layer() {",
            '    std::cout << "Executando camada Conv2D" << std::endl;',
            "    // TODO: Implementar convolução 2D",
            "}",
            "",
            "void avgpool2d_layer() {",
            '    std::cout << "Executando camada AvgPool2D" << std::endl;',
            "    // TODO: Implementar average pooling 2D",
            "}",
            "",
            "void lif_neuron_layer() {",
            '    std::cout << "Executando neurônios LIF" << std::endl;',
            "    // TODO: Implementar neurônios Leaky Integrate-and-Fire",
            "}",
            "",
            "void linear_layer() {",
            '    std::cout << "Executando camada Linear" << std::endl;',
            "    // TODO: Implementar camada totalmente conectada",
            "}",
            "",
            "void flatten_layer() {",
            '    std::cout << "Executando camada Flatten" << std::endl;',
            "    // TODO: Implementar flatten",
            "}",
            "",
            "void execute_network() {",
            '    std::cout << "=== Executando Rede Neural Spiking ===" << std::endl;',
            ""
        ]
        
        # Adiciona as chamadas das funções na ordem correta
        for node_name in self.execution_order:
            node = self.nodes[node_name]
            function_call = self._generate_function_call(node_name, node)
            implementation.append(function_call)
            implementation.append("")
        
        implementation.extend([
            '    std::cout << "=== Execução concluída ===" << std::endl;',
            "}"
        ])
        
        return "\n".join(implementation)
    
    def generate_main_cpp(self) -> str:
        """Gera um arquivo main.cpp de exemplo."""
        main_code = [
            '#include "nir_network.h"',
            "",
            "int main() {",
            "    execute_network();",
            "    return 0;",
            "}"
        ]
        return "\n".join(main_code)
    
    def print_network_info(self):
        """Imprime informações sobre a rede."""
        print("=== Informações da Rede Neural ===")
        print(f"Número de nós: {len(self.nodes)}")
        print(f"Número de arestas: {len(self.edges)}")
        print(f"Ordem de execução: {self.execution_order}")
        
        print("\n=== Tipos de Nós ===")
        for name, node in self.nodes.items():
            node_type = type(node).__name__
            print(f"{name}: {node_type}")
        
        print("\n=== Conectividade ===")
        for src, dst in self.edges:
            print(f"{src} -> {dst}")
    
    def export_weights_to_c_header(
        self,
        header_path: str = "nir_weights.h",
        ctype: str = "float",
        emit_typedef_if_builtin: bool = True,
        line_wrap: int = 10,
        float_fmt: str = ".8f",
        verbose: bool = True
    ):
        """
        Exporta pesos do arquivo NIR para um arquivo .h compatível com C.
        Segue o mesmo padrão do export_weights.py
        """
        if verbose:
            print("Extraindo pesos do arquivo NIR...")

        # Cria diretório se não existir
        header_dir = os.path.dirname(header_path)
        if header_dir and not os.path.exists(header_dir):
            os.makedirs(header_dir, exist_ok=True)

        # guard a partir do nome do arquivo
        base = os.path.basename(header_path)
        guard = re.sub(r"[^A-Za-z0-9]", "_", base).upper()

        def sanitize(k: str) -> str:
            # "layer1.0.conv1.weight" -> "layer1_0_conv1_weight"
            sanitized = re.sub(r"[^A-Za-z0-9_]", "_", k.replace(".", "_"))
            # Se começar com dígito, adiciona prefixo "layer_"
            if sanitized and sanitized[0].isdigit():
                sanitized = "layer_" + sanitized
            return sanitized

        def as_list_str(arr: np.ndarray) -> list[str]:
            # sempre grava como float no texto
            flat = arr.astype(np.float32).reshape(-1).tolist()
            fmt = "{:" + float_fmt + "}"
            return [fmt.format(v) for v in flat]

        def write_wrapped(f, values: Iterable[str], wrap=line_wrap, indent="  "):
            values = list(values)  # garantir len()
            for i, v in enumerate(values):
                if i % wrap == 0:
                    f.write("\n" + indent)
                f.write(v)
                if i != len(values) - 1:
                    f.write(",")
            f.write("\n")

        if verbose:
            print(f"Exportando pesos NIR para {header_path}...")

        stats = []
        weights_found = []

        # Coleta todos os pesos e biases dos nós
        for node_name, node in self.nodes.items():
            # Pula nós de entrada e saída
            if node_name in ['input', 'output']:
                continue
                
            # Extrai weights
            if hasattr(node, 'weight') and node.weight is not None:
                weight_name = f"{node_name}_weight"
                weights_found.append((weight_name, node.weight))
                
            # Extrai bias
            if hasattr(node, 'bias') and node.bias is not None:
                bias_name = f"{node_name}_bias"
                weights_found.append((bias_name, node.bias))
                
            # Extrai parâmetros específicos de neurônios LIF
            if hasattr(node, 'tau') and node.tau is not None:
                tau_name = f"{node_name}_tau"
                weights_found.append((tau_name, node.tau))
                
            if hasattr(node, 'v_threshold') and node.v_threshold is not None:
                thresh_name = f"{node_name}_v_threshold"
                weights_found.append((thresh_name, node.v_threshold))
                
            if hasattr(node, 'v_leak') and node.v_leak is not None:
                leak_name = f"{node_name}_v_leak"
                weights_found.append((leak_name, node.v_leak))

        with open(header_path, "w") as f:
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            # typedef opcional para weight_t
            if ctype in {"float", "double"} and emit_typedef_if_builtin:
                f.write(f"typedef {ctype} weight_t;\n\n")

            f.write(f"// Número de tensores exportados do NIR\n")
            f.write(f"#define NIR_NUM_TENSORS {len(weights_found)}\n\n")

            for weight_name, weight_array in weights_found:
                name = sanitize(weight_name)
                arr = np.array(weight_array, dtype=np.float32)
                dims = list(arr.shape)
                rank = arr.ndim

                # Metadados do shape
                if dims:
                    for i, d in enumerate(dims):
                        f.write(f"#define {name.upper()}_DIM{i} {d}\n")
                f.write(f"#define {name.upper()}_NDIMS {rank}\n\n")

                # Estatísticas
                mn = float(arr.min()) if arr.size > 0 else 0.0
                mx = float(arr.max()) if arr.size > 0 else 0.0
                stats.append((weight_name, mn, mx))

                # Emissão por rank
                if rank == 2:
                    rows, cols = dims
                    f.write(f"{ctype} {name}[{rows}][{cols}] = {{\n")
                    fmt = "{:" + float_fmt + "}"
                    for i in range(rows):
                        row = arr[i].reshape(-1).tolist()
                        f.write("  {")
                        for j, val in enumerate(row):
                            f.write(fmt.format(val))
                            if j != cols - 1:
                                f.write(",")
                        f.write("}")
                        if i != rows - 1:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.write("};\n\n")
                elif rank == 4:
                    # Caso especial para Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
                    out_ch, in_ch, kh, kw = dims
                    f.write(f"{ctype} {name}[{out_ch}][{in_ch}][{kh}][{kw}] = {{\n")
                    fmt = "{:" + float_fmt + "}"
                    
                    for out_idx in range(out_ch):
                        f.write("  {\n")
                        for in_idx in range(in_ch):
                            f.write("    {\n")
                            for h_idx in range(kh):
                                f.write("      {")
                                for w_idx in range(kw):
                                    val = arr[out_idx, in_idx, h_idx, w_idx]
                                    f.write(fmt.format(val))
                                    if w_idx != kw - 1:
                                        f.write(",")
                                f.write("}")
                                if h_idx != kh - 1:
                                    f.write(",\n")
                                else:
                                    f.write("\n")
                            f.write("    }")
                            if in_idx != in_ch - 1:
                                f.write(",\n")
                            else:
                                f.write("\n")
                        f.write("  }")
                        if out_idx != out_ch - 1:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.write("};\n\n")
                else:
                    n = arr.size
                    vals = as_list_str(arr)
                    f.write(f"{ctype} {name}[{n}] = {{")
                    write_wrapped(f, vals, wrap=line_wrap, indent="  ")
                    f.write("};\n\n")

            f.write(f"#endif // {guard}\n")

        if verbose:
            print("Pesos NIR exportados com sucesso!")
            print(f"Arquivo criado: {header_path}")
            print(f"Total de tensores exportados: {len(weights_found)}")
            for k, mn, mx in stats:
                print(f"  {k}: min={mn:{float_fmt}}, max={mx:{float_fmt}}")

    def save_cpp_files(self, output_dir: str = "."):
        """Salva os arquivos C++ gerados."""
        import os
        
        # Cria diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva header
        header_path = os.path.join(output_dir, "nir_network.h")
        with open(header_path, 'w') as f:
            f.write(self.generate_cpp_header())
        print(f"Header salvo em: {header_path}")
        
        # Salva implementação
        cpp_path = os.path.join(output_dir, "nir_network.cpp")
        with open(cpp_path, 'w') as f:
            f.write(self.generate_cpp_implementation())
        print(f"Implementação salva em: {cpp_path}")
        
        # Salva main
        main_path = os.path.join(output_dir, "main.cpp")
        with open(main_path, 'w') as f:
            f.write(self.generate_main_cpp())
        print(f"Main salvo em: {main_path}")


def export_nir_weights_to_c_header(
    nir_file: str,
    header_path: str = "nir_weights.h",
    ctype: str = "float",
    emit_typedef_if_builtin: bool = True,
    line_wrap: int = 10,
    float_fmt: str = ".8f",
    verbose: bool = True
):
    """
    Função standalone para exportar pesos de um arquivo NIR para C header.
    Similar à função export_weights_to_c_header_generic do export_weights.py
    
    Args:
        nir_file: Caminho para o arquivo .nir
        header_path: Caminho para o arquivo .h de saída
        ctype: Tipo C para os pesos (ex: "float", "double", "weight_t")
        emit_typedef_if_builtin: Se deve emitir typedef para tipos builtin
        line_wrap: Número de valores por linha
        float_fmt: Formato para números float
        verbose: Se deve imprimir informações detalhadas
    """
    parser = NIRToCppParser(nir_file)
    parser.export_weights_to_c_header(
        header_path=header_path,
        ctype=ctype,
        emit_typedef_if_builtin=emit_typedef_if_builtin,
        line_wrap=line_wrap,
        float_fmt=float_fmt,
        verbose=verbose
    )


def demo_simple_export():
    """Demonstração simples de exportação de pesos do NIR."""
    nir_file = "csnn_mnist.nir"
    
    if not os.path.exists(nir_file):
        print(f"Erro: Arquivo {nir_file} não encontrado!")
        print("Execute primeiro export_nir_from_snnTorchNet.py para gerar o arquivo NIR.")
        return
    
    print("=== DEMONSTRAÇÃO: Exportação Simples de Pesos do NIR ===")
    
    # Exportação simples
    export_nir_weights_to_c_header(
        nir_file=nir_file,
        header_path="nir_weights_simple.h",
        ctype="float",
        verbose=True
    )
    
    print("Demonstração concluída!")


def main():
    """Função principal."""
    nir_file = "csnn_mnist.nir"
    
    try:
        # Cria o parser
        parser = NIRToCppParser(nir_file)
        
        # Mostra informações da rede
        parser.print_network_info()
        
        print("\n" + "="*50)
        print("EXPORTAÇÃO DE PESOS DO NIR")
        print("="*50)
        
        # Exportação dos pesos para arquivo .h (dentro de cpp_output)
        export_nir_weights_to_c_header(
            nir_file=nir_file,
            header_path="cpp_output/nir_weights.h",
            ctype="float",
            emit_typedef_if_builtin=False,
            line_wrap=10,
            float_fmt=".8f",
            verbose=True
        )
        
        print("\n" + "="*50)
        print("Código C++ gerado:")
        print("="*50)
        
        # Mostra o código gerado
        print("\n--- nir_network.h ---")
        print(parser.generate_cpp_header())
        
        print("\n--- nir_network.cpp ---")
        print(parser.generate_cpp_implementation())
        
        print("\n--- main.cpp ---")
        print(parser.generate_main_cpp())
        
        # Salva os arquivos
        print("\n" + "="*50)
        parser.save_cpp_files("cpp_output")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo {nir_file} não encontrado!")
        print("Execute primeiro export_nir_from_snnTorchNet.py para gerar o arquivo NIR.")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    # Descomente a linha abaixo para executar apenas a demonstração simples
    # demo_simple_export()
    
    # Execução completa
    main()

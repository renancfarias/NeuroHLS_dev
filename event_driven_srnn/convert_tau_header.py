import sys
import os
import re
from calculate_tau_scaled import calculate_tau_scaled

def convert_header_file(input_filepath: str, fractional_bits: int = 16):
    if not os.path.exists(input_filepath):
        print(f"Erro: Arquivo {input_filepath} não encontrado.")
        return

    # Gera o nome do arquivo de saída
    base_name, ext = os.path.splitext(input_filepath)
    output_filepath = f"{base_name}_scaled{ext}"

    with open(input_filepath, 'r') as f:
        content = f.read()

    # Expressão regular para achar arrays C das variáveis tau_syn_x e tau_mem_x
    # Ex: float tau_syn_0[] = {1.0, 2.0}; ou auto tau_mem_1[4] = { 10.5, 2.0 };
    array_pattern = re.compile(
        r'((?:const\s+)?[\w_]+\s+)(tau_(?:syn|mem)_\d+)((?:\[.*?\])?\s*=\s*\{)([^}]+)(\}\s*;)',
        re.MULTILINE
    )

    def replace_tau_array(match):
        prefix = match.group(1)
        array_name = match.group(2)
        middle = match.group(3)
        array_values_str = match.group(4)
        suffix = match.group(5)
        
        # Extrai os floats removendo comentários/vírgulas/espaços
        str_values = [v.strip() for v in array_values_str.split(',') if v.strip()]
        scaled_values = []
        
        for val_str in str_values:
            # Tenta converter os valores pra float (caso haja sufixos como 'f')
            clean_val = val_str.replace('f', '').replace('F', '')
            try:
                tau_val = float(clean_val)
                scaled_val = calculate_tau_scaled(tau_val, fractional_bits)
                scaled_values.append(str(scaled_val))
            except ValueError:
                print(f"Aviso: Não foi possível converter o valor '{val_str}' em {array_name}.")
                scaled_values.append("0 /*ERRO*/")

        # Junta os novos valores
        new_values_str = ", ".join(scaled_values)
        
        # Mantém o tipo (prefix) original da variável em vez de forçar um novo
        return f"{prefix}{array_name}{middle}\n    {new_values_str}\n{suffix}"

    # Substitui os arrays e mantém todo o resto do conteúdo intocado
    modified_content = array_pattern.sub(replace_tau_array, content)
    
    # Adiciona o include se não existir
    if "#include <stdint.h>" not in modified_content and "decay_t" in modified_content:
        # Tenta colocar no começo do arquivo
        modified_content = "#include <stdint.h>\n" + modified_content

    # Salva o arquivo final
    with open(output_filepath, 'w') as f:
        f.write(modified_content)
        
    print(f"Sucesso! Arquivo gerado em: {output_filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convert_tau_header.py <caminho_para_arquivo_header.h> [bits_fracionarios]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    
    convert_header_file(input_file, bits)

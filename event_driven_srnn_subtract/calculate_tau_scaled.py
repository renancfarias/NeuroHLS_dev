import math

def calculate_tau_scaled(tau: float, fractional_bits: int = 16) -> int:
    """
    Calcula o valor inteiro escalonado de tau para o cálculo do decaimento 
    exponencial através do algoritmo TS-EFA usando matemática de Ponto Fixo (Fixed-Point).
    
    A fórmula é: round( (1 / (tau * ln(2))) * (2 ^ fractional_bits) )
    
    Args:
        tau (float): A constante de tempo (tau_mem ou tau_syn).
        fractional_bits (int): O número de bits dedicados à parte fracionária no C/C++ (Padrão: 16).
        
    Returns:
        int: O multiplicador inteiro que deve ser passado para a função C/C++.
    """
    if tau <= 0:
        raise ValueError("O valor de tau deve ser maior que zero.")
        
    ln2 = math.log(2)
    inv_tau_ln2 = 1.0 / (tau * ln2)
    scale_factor = 1 << fractional_bits # Equivalente a 2^fractional_bits
    
    # Aplica a escala e arredonda para retornar o valor inteiro
    tau_scaled = int(round(inv_tau_ln2 * scale_factor))
    return tau_scaled

if __name__ == "__main__":
    # Exemplo prático de uso antes de exportar os parâmetros pro código C++
    
    FRACTIONAL_BITS = 10
    
    # Valores de exemplo de tau da membrana e sinapse (em milissegundos ou na unidade de dt base)
    tau_mem = 20.0
    tau_syn = 5.0
    
    tau_mem_scaled = calculate_tau_scaled(tau_mem, FRACTIONAL_BITS)
    tau_syn_scaled = calculate_tau_scaled(tau_syn, FRACTIONAL_BITS)
    
    print(f"=== Conversor de Constantes de Tempo para TS-EFA Fixed-Point ===")
    print(f"Bits Fracionários do C++ (FRACTIONAL_BITS): {FRACTIONAL_BITS}\n")
    
    print(f"Para tau_mem = {tau_mem:>5.1f} -> {tau_mem_scaled}  (HEX: {hex(tau_mem_scaled)})")
    print(f"Para tau_syn = {tau_syn:>5.1f} -> {tau_syn_scaled} (HEX: {hex(tau_syn_scaled)})")
    
    print("\nVocê pode passar estes valores inteiros diretamente para as variáveis no cabeçalho do C/C++!")

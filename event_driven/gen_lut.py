import math

# Configurações
FILENAME = "exp_lut_values.h"
LUT_BITS = 8
NUM_ENTRIES = 2 ** LUT_BITS # 256 entradas

def generate_header():
    with open(FILENAME, "w") as f:
        f.write("// Arquivo gerado automaticamente via script Python\n")
        f.write(f"// Tabela de Lookup para 2^(-x) onde x varia de 0 a {NUM_ENTRIES-1}/{NUM_ENTRIES}\n")
        f.write("// Esses valores devem ser incluidos dentro da definicao do array estatico\n\n")
        
        # Loop para calcular cada entrada
        for i in range(NUM_ENTRIES):
            # O índice 'i' representa a fração f = i / 256
            # f varia de 0.0 (incluso) até 0.996 (aprox)
            fraction = i / float(NUM_ENTRIES)
            
            # Calculamos 2^(-fraction)
            # Math.pow(2, -x) é equivalente a exp(-x * ln(2))
            val = math.pow(2.0, -fraction)
            
            # Escreve o valor no arquivo
            # Usamos 6 casas decimais para garantir precisão na conversão do HLS
            if i < NUM_ENTRIES - 1:
                f.write(f"{val:.8f}, ")
            else:
                f.write(f"{val:.8f}  // Ultimo valor")
            
            # Quebra de linha para ficar legível (8 valores por linha)
            if (i + 1) % 8 == 0:
                f.write("\n")

    print(f"Sucesso! Arquivo '{FILENAME}' gerado com {NUM_ENTRIES} entradas.")

if __name__ == "__main__":
    generate_header()
from read_nir import read_nir
from implementation_manager import *

def teste(nir_file):
    model = read_nir(nir_file)
    print(model)

    # print("IMPLEMENTATION:\n\n")

    implement_model(model)


# teste("z_nir_examples/lif_norse.nir")
# teste("z_nir_examples/cnn_sinabs.nir")
teste("z_nir_examples/braille_noDelay_bias_zero.nir")
from types import SimpleNamespace

class LayerConfig:

    def __init__(self):

        self.accum_unroll_factor = 1
        self.fire_unroll_factor = 1

        self.quant_total_bits = 16
        self.quant_int_bits = 8

    def __str__(self):
        s = "  Unroll Factors:\n"
        s += f"    - Accum: {self.accum_unroll_factor}\n"
        s += f"    - Fire: {self.fire_unroll_factor}\n"

        s += f"  Quantization (ap_fixed<{self.quant_total_bits}, {self.quant_int_bits}>):\n"
        s += f"    - Total bits: {self.quant_total_bits}\n"
        s += f"    - Integer bits: {self.quant_int_bits}\n"
        s += f"    - Fractional bits: {self.quant_total_bits - self.quant_int_bits}\n"

        return s
    
    def set_potential_quantization(self, total_bits, int_bits):
        self.quant_total_bits = total_bits
        self.quant_int_bits = int_bits

    def set_unroll_factors(self, accum_unroll_factor, fire_unroll_factor):
        self.accum_unroll_factor = accum_unroll_factor
        self.fire_unroll_factor = fire_unroll_factor

class DenseLayerConfig(LayerConfig):

    def __init__(self, n_inputs: int, n_neurons: int):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def __str__(self):
        s = "-" * 30 + "\n"
        s += f"Dense ({self.n_inputs}, {self.n_neurons})\n\n"
        s += super().__str__()

        return s

class ModelConfigObjectCreator:
    
    def __init__(self):
        self.obj = SimpleNamespace()
        self.layer_count = 1

    # def __str__(self):
    #     s = ""

    #     for layer in self.layers:
    #         s += layer.__str__()
        
    #     return s
    
    def add_layer(self, layer: LayerConfig):

        if isinstance(layer, DenseLayerConfig):
            layer_name = f"layer_{self.layer_count}_DENSE_{layer.n_inputs}_{layer.n_neurons}"
            setattr(self.obj, layer_name, layer)

        self.layer_count += 1

    def get_object(self):
        return self.obj
    
# def test_model_config():

#     # Feito a partir da leitura do NIR

#     layer_1 = DenseLayerConfig(784, 128)
#     layer_2 = DenseLayerConfig(128, 10)

#     model = ModelConfigObjectCreator()
#     model.add_layer(layer_1)
#     model.add_layer(layer_2)

#     print(model)

#     # User pode definir as configurações

#     model.layers[0].set_unroll_factors(10, 10)
#     model.layers[1].set_potential_quantization(32, 4)
#     print(model.layers[0])
#     print(model.layers[1])

    

# test_model_config()
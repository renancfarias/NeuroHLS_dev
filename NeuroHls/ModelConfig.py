from typing import List
from .FileGenUtils import get_closest_divisor

class LayerConfig:

    def __init__(self):

        self._accum_unroll_factor = 1
        self._fire_unroll_factor = 1

        self._potentials_total_bits = 16
        self._potentials_int_bits = 8

    def __str__(self):
        s = "\tUnroll Factors:\n"
        s += f"\t\t- Accum: {self._accum_unroll_factor}\n"
        s += f"\t\t- Fire: {self._fire_unroll_factor}\n"

        s += f"\tQuantization (ap_fixed<{self._potentials_total_bits}, {self._potentials_int_bits}>):\n"
        s += f"\t\t- Total bits: {self._potentials_total_bits}\n"
        s += f"\t\t- Integer bits: {self._potentials_int_bits}\n"
        s += f"\t\t- Fractional bits: {self._potentials_total_bits - self._potentials_int_bits}\n"

        return s
    
    def set_potential_quantization(self, total_bits, int_bits):
        self._potentials_total_bits = total_bits
        self._potentials_int_bits = min(int_bits, total_bits)

    def set_unroll_factors(self, accum_unroll_factor, fire_unroll_factor):
        self._accum_unroll_factor = accum_unroll_factor
        self._fire_unroll_factor = fire_unroll_factor

class DenseLayerConfig(LayerConfig):

    def __init__(self, n_inputs: int, n_neurons: int):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def __str__(self):
        s = f"Dense ({self.n_inputs}, {self.n_neurons})\n"
        s += "-" * 30 + "\n\n"
        s += super().__str__()

        return s
    
    def set_unroll_factors(self, accum_unroll_factor, fire_unroll_factor, verbose=True):

        correct_accum_unroll_factor = get_closest_divisor(self.n_inputs, accum_unroll_factor)
        correct_fire_unroll_factor = get_closest_divisor(self.n_neurons, fire_unroll_factor)

        super().set_unroll_factors(correct_accum_unroll_factor, correct_fire_unroll_factor)

        if not verbose:
            return

        if correct_accum_unroll_factor != accum_unroll_factor:
            print(f"The accumulation unroll factor must be a divisor of the number of inputs ({self.n_inputs}). Used value: {correct_accum_unroll_factor}")
        else:
            print(f"Used accumulation unroll factor: {correct_accum_unroll_factor}")

        if correct_fire_unroll_factor != fire_unroll_factor:
            print(f"The fire unroll factor must be a divisor of the number of neurons ({self.n_neurons}). Used value: {correct_fire_unroll_factor}")
        else:
            print(f"Used fire unroll factor: {correct_fire_unroll_factor}")

    def get_input_shape(self):

        return (self.n_inputs, )
    
    def get_output_shape(self):
        
        return (self.n_neurons, )
    
class ModelConfig:
    
    def __init__(self):

        self.layers: List[LayerConfig] = []

        self.input_total_bits = 16
        self.input_int_bits = 8

    def __str__(self):
        
        s = ""
        for idx, layer in enumerate(self.layers):

            if idx > 0:
                s += "\n"

            s += 30 * "-" + "\n"
            s += f"Layer {idx+1}: {layer.__str__()}"
        
        return s
    
    def add_layer(self, layer: LayerConfig):

        self.layers.append(layer)

    def set_input_quantization(self, total_bits, int_bits):

        self.input_total_bits = total_bits
        self.input_int_bits = int_bits

    def set_default_potential_quantization(self, total_bits, int_bits):

        self.input_total_bits = total_bits
        self.input_int_bits = int_bits

        for layer in self.layers:
            layer.set_potential_quantization(total_bits, int_bits)

    def set_default_unroll_factors(self, accum_unroll_factor, fire_unroll_factor):

        for layer in self.layers:
            layer.set_unroll_factors(accum_unroll_factor, fire_unroll_factor)
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neuro_hls.implementation_manager.implement_model import implement_model
from neuro_hls.read_nir.layer_configuration import IF, Input, Linear, ModelConfig, Output


BUILD = Path("/tmp/neurohls_recurrent_csim")


def create_model():
    model = ModelConfig()
    layers = [
        Input("input", np.array([1])),
        Linear("feed", (1,), (1,), np.array([[1.0]])),
        IF(
            "neuron", (1,), (1,), np.array([1.0]),
            np.array([0.5]), np.array([0.0])
        ),
        Linear("recurrent", (1,), (1,), np.array([[1.0]])),
        Output("output", np.array([1])),
    ]
    model.graph_layers = {layer.name: layer for layer in layers}
    model.graph_edges = [
        ("input", "feed"),
        ("feed", "neuron"),
        ("neuron", "recurrent"),
        ("recurrent", "neuron"),
        ("neuron", "output"),
    ]
    model.input_shape = np.array([1])
    model.output_shape = np.array([1])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", action="store_true", help="also run HLS synthesis")
    args = parser.parse_args()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    implement_model(create_model(), BUILD, use_float=True, use_event_driven=True)

    testbench = BUILD / "testbench.cpp"
    testbench.write_text(
        '#include <cassert>\n#include "snn_implementation.h"\n'
        'int main() { float active[1] = {1}; float silent[1] = {0}; '
        'bit_t output[1] = {}; snn_to_hls(active, output, true); '
        'assert(output[0] == 1); output[0] = 0; '
        'snn_to_hls(silent, output, false); assert(output[0] == 1); return 0; }\n'
    )

    project = BUILD / "project"
    tcl = BUILD / "run.tcl"
    synthesis_command = "csynth_design\n" if args.synth else ""
    tcl.write_text(
        'open_project -reset project\n'
        'set_top snn_to_hls\n'
        f'add_files -cflags "-I{ROOT / "neuro_hls" / "backend"} -I{BUILD}" snn_implementation.cpp\n'
        f'add_files -tb -cflags "-I{ROOT / "neuro_hls" / "backend"} -I{BUILD}" testbench.cpp\n'
        'open_solution -reset sol -flow_target vitis\n'
        'set_part {xcu250-figd2104-2L-e}\n'
        'create_clock -period 10\n'
        'csim_design\n'
        f'{synthesis_command}'
        'exit\n'
    )
    subprocess.run(
        ["vitis-run", "--mode", "hls", "--tcl", str(tcl)],
        cwd=BUILD,
        check=True,
    )


if __name__ == "__main__":
    main()

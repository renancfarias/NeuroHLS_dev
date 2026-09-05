import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nir

from neuro_hls.implementation_manager.implement_model import implement_model
from neuro_hls.read_nir.get_model_config_from_nir import get_model_config_from_nir
from neuro_hls.read_nir.layer_configuration import (
    AvgPool2d,
    Conv1d,
    Conv2d,
    Input,
    IF,
    LIF,
    Linear,
    ModelConfig,
    Output,
    Scale,
)


ROOT = Path(__file__).resolve().parents[1]


class P1FrontendTests(unittest.TestCase):

    def test_lif_time_constants_use_high_resolution_temporal_type(self):
        model = ModelConfig()
        model.add_layer(Input("input", np.array([2])))
        neuron = LIF(
            "lif", (2,), (2,),
            np.full(2, 2e-4), np.full(2, 2.0), np.zeros(2),
            np.ones(2), np.zeros(2),
        )
        neuron.add_dependency("input", is_recurrent=False)
        model.add_layer(neuron)
        output = Output("output", np.array([2]))
        output.add_dependency("lif", is_recurrent=False)
        model.add_layer(output)

        for backend in ("time-driven", "event-driven"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                build_dir = Path(temporary)
                implement_model(model, build_dir, use_float=False, backend=backend)
                params = (build_dir / "neuron_params.h").read_text(encoding="utf-8")
                quantization = (build_dir / "quantization.h").read_text(encoding="utf-8")
                self.assertIn("ap_fixed<32, 8, AP_RND> temporal_t", quantization)
                expected_type = "event_tau_t" if backend == "event-driven" else "temporal_t"
                self.assertIn(f"{expected_type} tau_", params)
                self.assertNotIn("weight_t tau_", params)

    def test_conv1d_normalizes_same_and_omits_optional_bias(self):
        layer = Conv1d(
            "conv",
            input_shape=(2, 5),
            output_shape=(4, 5),
            weight=np.ones((4, 2, 3)),
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=None,
        )

        self.assertEqual(layer.padding, (1,))
        self.assertEqual(
            layer.get_template_args(),
            {
                "kernel": (3,),
                "stride": (1,),
                "padding": (1,),
                "dilation": (1,),
                "groups": 1,
            },
        )
        self.assertEqual(tuple(layer.get_neuron_params()), ("weights",))

    def test_conv1d_rejects_asymmetric_same_padding(self):
        with self.assertRaisesRegex(ValueError, "asymmetric padding"):
            Conv1d(
                "conv",
                input_shape=(1, 5),
                output_shape=(1, 5),
                weight=np.ones((1, 1, 2)),
                stride=1,
                padding="same",
                dilation=1,
                groups=1,
                bias=None,
            )

    def test_conv2d_valid_padding_and_optional_bias(self):
        layer = Conv2d(
            "conv",
            input_shape=(1, 4, 4),
            output_shape=(2, 2, 2),
            weight=np.ones((2, 1, 3, 3)),
            stride=(1, 1),
            padding="valid",
            dilation=(1, 1),
            groups=1,
            bias=None,
        )

        self.assertEqual(layer.padding, (0, 0))
        self.assertNotIn("bias", layer.get_neuron_params())

    def test_avg_pool_exposes_static_template_arguments(self):
        layer = AvgPool2d(
            "pool",
            input_shape=(2, 4, 4),
            output_shape=(2, 2, 2),
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
        )
        self.assertEqual(
            layer.get_template_args(),
            {
                "kernel": (2, 2),
                "stride": (2, 2),
                "padding": (0, 0),
            },
        )

    def test_scale_expands_scalar_for_static_broadcast(self):
        for shape in ((4,), (2, 3), (2, 2, 2)):
            with self.subTest(shape=shape):
                layer = Scale("scale", shape, shape, np.array(2.5))
                self.assertTrue(layer.scalar_broadcast)
                self.assertEqual(layer.scale.shape, shape)
                np.testing.assert_array_equal(layer.scale, np.full(shape, 2.5))

    def test_scale_is_read_from_a_valid_nir_graph(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            nir_path = Path(temporary_directory) / "scale.nir"
            graph = nir.NIRGraph(
                nodes={
                    "input": nir.Input(input_type={"input": np.array([2, 3])}),
                    "scale": nir.Scale(scale=np.full((2, 3), 2.0)),
                    "output": nir.Output(output_type={"output": np.array([2, 3])}),
                },
                edges=[("input", "scale"), ("scale", "output")],
            )
            nir.write(str(nir_path), graph)

            model = get_model_config_from_nir(str(nir_path))
            scale_layer = next(layer for layer in model.layers if isinstance(layer, Scale))
            self.assertEqual(tuple(scale_layer.input_shape), (2, 3))
            np.testing.assert_array_equal(scale_layer.scale, np.full((2, 3), 2.0))

    def test_generator_emits_p1_calls_and_omits_conv_bias(self):
        layers_and_calls = [
            (
                Conv1d("op", (1, 5), (1, 5), np.ones((1, 1, 3)), 1, "same", 1, 1, None),
                "Conv1dReuse<3,1,1,1,1,1>(input, output, weights_1);",
            ),
            (
                Conv2d("op", (1, 4, 4), (1, 2, 2), np.ones((1, 1, 3, 3)), 1, "valid", 1, 1, None),
                "Conv2dReuse<3,3,1,1,0,0,1,1,1,1>(input, output, weights_1);",
            ),
            (
                AvgPool2d("op", (1, 4, 4), (1, 2, 2), 2, 2, 0),
                "AvgPool2dReuse<2,2,2,2,0,0,1>(input, output);",
            ),
            (
                Scale("op", (3,), (3,), np.array([1.0, 2.0, 3.0])),
                "Scale<3,1>(input, output, scale_1);",
            ),
        ]

        for layer, expected_call in layers_and_calls:
            with self.subTest(layer=type(layer).__name__):
                model = ModelConfig()
                model.add_layer(Input("input", layer.input_shape))
                layer.add_dependency("input", is_recurrent=False)
                model.add_layer(layer)
                output = Output("output", layer.output_shape)
                output.add_dependency(layer.name, is_recurrent=False)
                model.add_layer(output)

                with tempfile.TemporaryDirectory() as temporary_directory:
                    implement_model(model, temporary_directory, use_float=True)
                    generated = (Path(temporary_directory) / "snn_implementation.cpp").read_text()
                    self.assertIn(expected_call, generated)
                    if isinstance(layer, (Conv1d, Conv2d)):
                        self.assertNotIn("bias_1", generated)

    def test_time_driven_p1_primitives_compile_and_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "p1_time_driven_test"
            compile_result = subprocess.run(
                [
                    "g++",
                    "-std=c++11",
                    "-Wno-unknown-pragmas",
                    "-I",
                    str(ROOT / "tests" / "stubs"),
                    "-I",
                    str(ROOT / "neuro_hls" / "backend"),
                    str(ROOT / "tests" / "cpp" / "p1_time_driven_test.cpp"),
                    "-o",
                    str(executable),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_time_driven_is_the_canonical_backend_name(self):
        model = ModelConfig()
        model.add_layer(Input("input", np.array([1])))
        neuron = IF(
            "if", (1,), (1,), np.ones(1), np.ones(1), np.zeros(1)
        )
        neuron.add_dependency("input", is_recurrent=False)
        model.add_layer(neuron)
        output = Output("output", np.array([1]))
        output.add_dependency("if", is_recurrent=False)
        model.add_layer(output)

        with tempfile.TemporaryDirectory() as temporary_directory:
            implement_model(
                model, temporary_directory, use_float=True,
                backend="time-driven",
            )
            generated = (
                Path(temporary_directory) / "snn_implementation.cpp"
            ).read_text()
            self.assertIn(
                '#include "neuro_hls_functions/time_driven.h"', generated
            )
            self.assertNotIn(
                '#include "neuro_hls_functions/dense.h"', generated
            )

        with self.assertRaisesRegex(ValueError, "backend must be"):
            implement_model(model, "/tmp/unused-neuro-hls", backend="parallel")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            implement_model(
                model, "/tmp/unused-neuro-hls", use_event_driven=True,
                backend="time-driven",
            )

    def test_event_driven_p1_primitives_compile_and_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "p1_event_driven_test"
            compile_result = subprocess.run(
                [
                    "g++",
                    "-std=c++11",
                    "-Wno-unknown-pragmas",
                    "-I",
                    str(ROOT / "tests" / "stubs"),
                    "-I",
                    str(ROOT / "neuro_hls" / "backend"),
                    str(ROOT / "tests" / "cpp" / "p1_event_driven_test.cpp"),
                    "-o",
                    str(executable),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_event_driven_generator_compiles_and_runs_sequential_model(self):
        model = ModelConfig()
        model.add_layer(Input("input", np.array([2])))
        linear = Linear("linear", (2,), (2,), np.eye(2))
        linear.add_dependency("input", is_recurrent=False)
        model.add_layer(linear)
        neuron = IF(
            "if", (2,), (2,), np.ones(2), np.ones(2), np.zeros(2)
        )
        neuron.add_dependency("linear", is_recurrent=False)
        model.add_layer(neuron)
        output = Output("output", np.array([2]))
        output.add_dependency("if", is_recurrent=False)
        model.add_layer(output)

        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(model, build_dir, use_float=True, use_event_driven=True)
            driver = build_dir / "driver.cpp"
            driver.write_text(
                '#include <cassert>\n#include "snn_implementation.h"\n'
                'int main() { hls::stream<ed_spike_t> input, output; '
                'ed_spike_t event = {}; event.type = ED_TYPE_SPIKE; '
                'event.amplitude = 2; event.width_idx = 0; input.write(event); '
                'event = {}; event.type = ED_TYPE_END_STEP; input.write(event); '
                'snn_to_hls(input, output, true); '
                'bool spike0 = false, spike1 = false; while (true) { '
                'event = output.read(); if (event.type != ED_TYPE_SPIKE) break; '
                'if (event.width_idx == 0) spike0 = true; '
                'if (event.width_idx == 1) spike1 = true; } '
                'assert(spike0); assert(!spike1); return 0; }\n'
            )
            executable = build_dir / "generated_event_model"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(build_dir),
                    str(build_dir / "snn_implementation.cpp"), str(driver),
                    "-o", str(executable),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)


if __name__ == "__main__":
    unittest.main()

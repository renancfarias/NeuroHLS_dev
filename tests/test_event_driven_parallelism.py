import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from neuro_hls.implementation_manager.implement_model import implement_model
from neuro_hls.read_nir.layer_configuration import (
    Conv2d,
    IF,
    Input,
    Linear,
    ModelConfig,
    Output,
)


ROOT = Path(__file__).resolve().parents[1]


class EventDrivenScalarTests(unittest.TestCase):
    def make_linear_model(self):
        model = ModelConfig()
        model.add_layer(Input("input", (5,)))
        layer = Linear("linear", (5,), (3,), np.ones((3, 5)))
        layer.add_dependency("input", False)
        model.add_layer(layer)
        neuron = IF(
            "if", (3,), (3,), np.ones(3), np.ones(3), np.zeros(3)
        )
        neuron.add_dependency("linear", False)
        model.add_layer(neuron)
        output = Output("output", (3,))
        output.add_dependency("if", False)
        model.add_layer(output)
        return model

    def test_nonzero_event_parallelism_is_rejected(self):
        model = self.make_linear_model()
        with self.assertRaisesRegex(ValueError, "removed"):
            model.define_event_driven_parallelism(0.5)
        with self.assertRaisesRegex(ValueError, "removed"):
            model.define_event_layer_parallelism("linear", 0.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.define_event_driven_parallelism(0.0)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_codegen_uses_scalar_actor_templates_and_keeps_dataflow(self):
        model = self.make_linear_model()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            implement_model(model, directory, use_float=True, backend="event-driven")
            source = (directory / "snn_implementation.cpp").read_text(encoding="utf-8")
            result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(directory),
                    "-c", str(directory / "snn_implementation.cpp"),
                    "-o", str(directory / "snn_implementation.o"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertIn("#pragma HLS DATAFLOW", source)
        self.assertIn("LinearSparse<5,3,15>", source)
        self.assertIn("IF<3,2>", source)
        self.assertNotIn("LinearSparse<5,3,15,", source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conv2d_has_no_configurable_lane_argument(self):
        model = ModelConfig()
        model.add_layer(Input("input", (1, 3, 3)))
        conv = Conv2d(
            "conv", (1, 3, 3), (4, 2, 2),
            np.ones((4, 1, 2, 2)), 1, "valid", 1, 1, np.zeros(4),
        )
        conv.add_dependency("input", False)
        model.add_layer(conv)
        neuron = IF(
            "if", (4, 2, 2), (4, 2, 2),
            np.ones((4, 2, 2)), np.ones((4, 2, 2)),
            np.zeros((4, 2, 2)),
        )
        neuron.add_dependency("conv", False)
        model.add_layer(neuron)
        output = Output("output", (4, 2, 2))
        output.add_dependency("if", False)
        model.add_layer(output)

        with tempfile.TemporaryDirectory() as directory:
            implement_model(model, Path(directory), use_float=True, backend="event-driven")
            source = (Path(directory) / "snn_implementation.cpp").read_text(encoding="utf-8")

        self.assertIn("Conv2d<2,2,1,1,0,0,1,1,1,1,3,3,4>", source)
        self.assertNotIn("Conv2d<2,2,1,1,0,0,1,1,1,1,3,3,4,", source)

    def test_runtime_has_no_lane_parameter(self):
        runtime = (ROOT / "neuro_hls" / "backend" / "neuro_hls_functions" / "event_driven.h").read_text(encoding="utf-8")
        self.assertNotIn("LANES", runtime)
        self.assertNotRegex(runtime, r"\bfor\s*\([^\n]*\blane\b")
        self.assertNotRegex(runtime, r"\bfor\s*\([^\n]*\bbase\b")
        self.assertNotIn("ARRAY_PARTITION variable=accum cyclic factor=1", runtime)


if __name__ == "__main__":
    unittest.main()

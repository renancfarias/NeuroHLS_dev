import json
import math
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from neuro_hls.implementation_manager.implement_model import implement_model
from neuro_hls.read_nir.layer_configuration import (
    Affine,
    AvgPool2d,
    Conv1d,
    Conv2d,
    CubaLIF,
    Flatten,
    IF,
    Input,
    LIF,
    Linear,
    Merge,
    ModelConfig,
    Output,
    Scale,
    SumPool2d,
)


ROOT = Path(__file__).resolve().parents[1]


def _ones(shape):
    return np.ones(shape, dtype=np.float64)


def _zeros(shape):
    return np.zeros(shape, dtype=np.float64)


def _plan_cases():
    shape = (5,)
    return [
        ("merge", Merge("merge", "left", "right", shape), 5, "elementwise"),
        ("flatten", Flatten("flatten", (1, 1, 5), shape), 5, "elementwise"),
        ("scale", Scale("scale", shape, shape, _ones(shape)), 5, "elementwise"),
        ("linear", Linear("linear", (5,), (3,), _ones((3, 5))), 15, "mac"),
        ("affine", Affine("affine", _ones((3, 5)), _zeros(3)), 15, "mac"),
        (
            "conv1d",
            Conv1d(
                "conv1d", (2, 5), (4, 5), _ones((4, 2, 3)),
                1, "same", 1, 1, None,
            ),
            120,
            "conv1d_mac",
        ),
        (
            "conv2d",
            Conv2d(
                "conv2d", (2, 4, 4), (3, 2, 2), _ones((3, 2, 3, 3)),
                1, "valid", 1, 1, None,
            ),
            216,
            "conv2d_mac",
        ),
        (
            "sum_pool",
            SumPool2d("sum_pool", (2, 5, 5), (2, 2, 2), 3, 2, 0),
            72,
            "pool_accumulate",
        ),
        (
            "avg_pool",
            AvgPool2d("avg_pool", (2, 5, 5), (2, 2, 2), 3, 2, 0),
            72,
            "pool_accumulate",
        ),
        ("if", IF("if", shape, shape, _ones(shape), _ones(shape), _zeros(shape)), 5, "neuron_update"),
        (
            "lif",
            LIF(
                "lif", shape, shape, _ones(shape), _ones(shape), _zeros(shape),
                _ones(shape), _zeros(shape),
            ),
            5,
            "neuron_update",
        ),
        (
            "cuba_lif",
            CubaLIF(
                "cuba_lif", shape, shape, _ones(shape), _ones(shape),
                _ones(shape), _zeros(shape), _ones(shape), _zeros(shape),
                _ones(shape),
            ),
            5,
            "neuron_update",
        ),
    ]


def _model_with_layer(layer):
    model = ModelConfig()
    model.add_layer(Input("input", layer.input_shape))
    layer.add_dependency("input", is_recurrent=False)
    model.add_layer(layer)

    if layer.emits_spike:
        previous = layer
    else:
        shape = tuple(int(value) for value in layer.output_shape)
        previous = IF(
            "after_layer", shape, shape, _ones(shape), _ones(shape),
            _zeros(shape),
        )
        previous.add_dependency(layer.name, is_recurrent=False)
        model.add_layer(previous)

    output = Output("output", previous.output_shape)
    output.add_dependency(previous.name, is_recurrent=False)
    model.add_layer(output)
    return model


class TimeDrivenParallelismPlanTests(unittest.TestCase):
    def test_rejects_non_finite_and_out_of_range_values(self):
        invalid = (-0.01, 1.01, math.nan, math.inf, -math.inf, True, False, None, "half")
        for value in invalid:
            with self.subTest(scope="model", value=value):
                with self.assertRaises(ValueError):
                    ModelConfig().define_time_driven_parallelism(value)
            with self.subTest(scope="layer", value=value):
                with self.assertRaises(ValueError):
                    Scale("scale", (5,), (5,), _ones(5)).define_time_driven_parallelism(value)

    def test_endpoints_resolve_the_documented_static_work_domain(self):
        for name, layer, work_items, operation_kind in _plan_cases():
            model = ModelConfig()
            for requested, expected_pe, expected_reuse in (
                (0.0, 1, work_items),
                (1.0, work_items, 1),
            ):
                with self.subTest(layer=name, requested=requested):
                    model.define_time_driven_parallelism(requested)
                    plan = model.resolve_time_driven_parallelism(layer)
                    self.assertEqual(requested, plan.requested_parallelism)
                    self.assertEqual(work_items, plan.total_work_items)
                    self.assertEqual(expected_pe, plan.processing_elements)
                    self.assertEqual(expected_reuse, plan.reuse_cycles)
                    self.assertEqual(0, plan.idle_slots)
                    self.assertEqual(operation_kind, plan.operation_kind)
                    self.assertEqual(expected_pe / work_items, plan.effective_parallelism)

    def test_fraction_rounds_half_up_and_records_tail_slots(self):
        layer = Linear("linear", (5,), (3,), _ones((3, 5)))
        plan = ModelConfig()
        plan.define_time_driven_parallelism(0.5)
        resolved = plan.resolve_time_driven_parallelism(layer)

        self.assertEqual(15, resolved.total_work_items)
        self.assertEqual(8, resolved.processing_elements)
        self.assertEqual(2, resolved.reuse_cycles)
        self.assertEqual(1, resolved.idle_slots)
        self.assertEqual(8 / 15, resolved.effective_parallelism)

    def test_small_positive_values_and_larger_values_are_monotonic(self):
        layer = Linear("linear", (5,), (3,), _ones((3, 5)))
        requested_values = (0.0, 1 / 15, 2 / 15, 0.25, 0.5, 1.0)
        resolved_units = []
        for requested in requested_values:
            model = ModelConfig()
            model.define_time_driven_parallelism(requested)
            resolved_units.append(
                model.resolve_time_driven_parallelism(layer).processing_elements
            )

        self.assertEqual([1, 1, 2, 4, 8, 15], resolved_units)
        self.assertEqual(sorted(resolved_units), resolved_units)

    def test_layer_override_wins_and_legacy_alias_warns(self):
        model = ModelConfig()
        inherited = Scale("inherited", (5,), (5,), _ones(5))
        overridden = Scale("overridden", (5,), (5,), _ones(5))
        model.add_layer(inherited)
        model.add_layer(overridden)
        model.define_time_driven_parallelism(0.25)
        model.define_time_driven_layer_parallelism("overridden", 0.75)

        self.assertEqual(1, model.resolve_time_driven_parallelism(inherited).processing_elements)
        self.assertEqual(4, model.resolve_time_driven_parallelism(overridden).processing_elements)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.define_layer_parallelism("inherited", 1.0)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        self.assertEqual(5, model.resolve_time_driven_parallelism(inherited).processing_elements)

    def test_reduction_toggle_was_removed(self):
        self.assertFalse(hasattr(ModelConfig(), "define_time_driven_reduction_parallelism"))

    def test_event_parallelism_nonzero_is_rejected(self):
        model = ModelConfig()
        with self.assertRaisesRegex(ValueError, "removed"):
            model.define_event_driven_parallelism(0.25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.define_event_driven_parallelism(0.0)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))


class TimeDrivenCodeGenerationTests(unittest.TestCase):
    def test_dense_generation_uses_flat_reuse_kernel_and_manifest(self):
        layer = Linear("linear", (5,), (3,), _ones((3, 5)))
        model = _model_with_layer(layer)
        model.define_time_driven_parallelism(0.5)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            implement_model(model, directory, use_float=True, backend="time-driven")
            source = (directory / "snn_implementation.cpp").read_text(encoding="utf-8")
            manifest = json.loads((directory / "parallelism_manifest.json").read_text(encoding="utf-8"))

        self.assertIn("LinearReuse<8>", source)
        self.assertNotIn("Linear<2,1>", source)
        record = next(item for item in manifest["layers"] if item["name"] == "linear")
        self.assertEqual("percent_parallel_reuse_v1", manifest["parallelism_contract"])
        self.assertEqual(15, record["total_work_items"])
        self.assertEqual(8, record["processing_elements"])
        self.assertEqual(2, record["reuse_cycles"])
        self.assertEqual(1, record["idle_slots"])

    def test_reduction_operators_generate_reuse_kernels(self):
        cases = [
            (
                Conv1d("op", (2, 5), (4, 5), _ones((4, 2, 3)), 1, "same", 1, 1, None),
                "Conv1dReuse<3,1,1,1,1,60>",
            ),
            (
                Conv2d("op", (2, 4, 4), (3, 2, 2), _ones((3, 2, 3, 3)), 1, "valid", 1, 1, None),
                "Conv2dReuse<3,3,1,1,0,0,1,1,1,108>",
            ),
            (
                SumPool2d("op", (2, 5, 5), (2, 2, 2), 3, 2, 0),
                "SumPool2dReuse<3,3,2,2,0,0,36>",
            ),
            (
                AvgPool2d("op", (2, 5, 5), (2, 2, 2), 3, 2, 0),
                "AvgPool2dReuse<3,3,2,2,0,0,36>",
            ),
        ]
        for layer, expected in cases:
            with self.subTest(layer=type(layer).__name__), tempfile.TemporaryDirectory() as directory:
                model = _model_with_layer(layer)
                model.define_time_driven_parallelism(0.5)
                implement_model(model, directory, use_float=True, backend="time-driven")
                source = (Path(directory) / "snn_implementation.cpp").read_text(encoding="utf-8")
                self.assertIn(expected, source)

    def test_generated_time_driven_models_compile(self):
        layers = [
            Affine("affine", _ones((3, 5)), _zeros(3)),
            Conv1d("conv", (2, 5), (4, 5), _ones((4, 2, 3)), 1, "same", 1, 1, None),
            Conv2d("conv", (2, 4, 4), (3, 2, 2), _ones((3, 2, 3, 3)), 1, "valid", 1, 1, None),
            SumPool2d("pool", (2, 5, 5), (2, 2, 2), 3, 2, 0),
            AvgPool2d("avg_pool", (2, 5, 5), (2, 2, 2), 3, 2, 0),
        ]
        for layer in layers:
            with self.subTest(layer=type(layer).__name__), tempfile.TemporaryDirectory() as directory:
                model = _model_with_layer(layer)
                model.define_time_driven_parallelism(0.5)
                implement_model(model, directory, use_float=True, backend="time-driven")
                result = subprocess.run(
                    [
                        "g++", "-std=c++11", "-Wno-unknown-pragmas",
                        "-I", str(ROOT / "tests" / "stubs"),
                        "-I", str(ROOT / "neuro_hls" / "backend"),
                        "-I", directory,
                        "-c", str(Path(directory) / "snn_implementation.cpp"),
                        "-o", str(Path(directory) / "snn_implementation.o"),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class TimeDrivenCppTests(unittest.TestCase):
    def test_percent_parallel_reuse_primitives_match_the_serial_reference(self):
        source = ROOT / "tests" / "cpp" / "time_driven_parallelism_test.cpp"
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "time_driven_parallelism_test"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    str(source), "-o", str(executable),
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


if __name__ == "__main__":
    unittest.main()

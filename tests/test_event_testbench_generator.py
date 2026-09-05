import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuro_hls.testbench_manager.testbench_manager import TestbenchManager


class EventTestbenchGeneratorTests(unittest.TestCase):

    def test_event_testbench_uses_zero_based_indices_and_stream_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            dataset = build_dir / "dataset.npz"
            np.savez(
                dataset,
                data=np.zeros((1, 2, 12), dtype=np.int32),
                labels=np.zeros((1,), dtype=np.int32),
            )

            manager = TestbenchManager(str(build_dir))
            manager.define_dataset(
                str(dataset), data_is_binary=True, step_count=2,
                different_sample_per_step=True
            )
            manager.define_sample_count_and_batch_size(1, 1)
            manager.create_testbench_file(
                (12,), 7, reset_potentials_between_inferences=True,
                use_event_driven=True
            )

            testbench = (build_dir / "testbench.cpp").read_text()
            self.assertIn("input_data[b][s][0]", testbench)
            self.assertIn("input_data[b][s][11]", testbench)
            self.assertNotIn("input_data[b][s][12]", testbench)
            self.assertIn("!= input_t(0)", testbench)
            self.assertIn("input_stream.write(event);", testbench)
            self.assertIn("output_stream.read();", testbench)
            self.assertIn(
                "s == STEP_COUNT - 1 ? ED_TYPE_END_SAMPLE : ED_TYPE_END_STEP",
                testbench,
            )
            self.assertIn(
                "(s + 1) * NEURO_HLS_EVENT_DT", testbench
            )
            self.assertNotIn("bit_t output[OUTPUT_SIZE]", testbench)
            self.assertNotIn("accum_output[i] += output[i]", testbench)


if __name__ == "__main__":
    unittest.main()

"""Shared, executable support for the NIR paper validation notebooks.

The notebooks deliberately stop at Vitis HLS C simulation.  They compare
functional behaviour, not RTL latency, resources, or power.  Generated HLS
projects and tool runs live below ``validation/build`` and ``validation/runs``
and are reproducible from the versioned notebooks and source artefacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import textwrap
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Matplotlib otherwise tries to create a cache below ~/.config, which is not
# guaranteed to be writable on batch/CI hosts.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALIDATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_ROOT.parent
BUILD_ROOT = VALIDATION_ROOT / "build"
RUNS_ROOT = VALIDATION_ROOT / "runs"
RESULTS_ROOT = VALIDATION_ROOT / "results"
CACHE_ROOT = VALIDATION_ROOT / ".cache"
NIR_EXAMPLES = REPO_ROOT / "nir_examples"
NMNIST_ROOT = REPO_ROOT / "nmnist"

PAPER_TAG = "v1.0.4"
PAPER_COMMIT = "8f9177e9f74374de7bd72e49b42217a24df716ec"
PAPER_DOI = "10.1038/s41467-024-52259-9"
PAPER_URL = "https://www.nature.com/articles/s41467-024-52259-9"
PAPER_REPOSITORY = "https://github.com/neuromorphs/NIR"
PROBE_SCHEMA_VERSION = 2
RAW_PAPER_BASE = (
    "https://raw.githubusercontent.com/neuromorphs/NIR/"
    f"{PAPER_TAG}/paper"
)

LIF_BASE_PATTERN = np.asarray(
    [
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    dtype=np.int8,
)

LIF_REFERENCE_FILES = {
    "Exact": "01_lif/lif_exact.csv",
    "Lava CPU fixed": "01_lif/lif_lava_cpu_fixed.csv",
    "Lava CPU float": "01_lif/lif_lava_cpu_float.csv",
    "Lava Loihi 2": "01_lif/lif_lava_loihi_fixed.csv",
    "Nengo": "01_lif/lif_nengo.csv",
    "Norse": "01_lif/lif_norse.csv",
    "Rockpool": "01_lif/lif_rockpool.csv",
    "Sinabs": "01_lif/lif_sinabs.csv",
    "snnTorch": "01_lif/lif_snntorch.csv",
    "SpiNNaker2": "01_lif/lif_spinnaker2.csv",
    "Spyx": "01_lif/lif_spyx.csv",
}

SCNN_ACTIVITY_FILES = {
    "Lava": "02_cnn/Lava_activity.npy",
    "Nengo": "02_cnn/nengo_activity.npy",
    "Norse": "02_cnn/Norse_activity.npy",
    "Sinabs": "02_cnn/sinabs_activity.npy",
    "snnTorch": "02_cnn/snnTorch_activity.npy",
    "Speck": "02_cnn/speck_activity.npy",
    "SpiNNaker2": "02_cnn/s2_activity.npy",
    "Spyx": "02_cnn/spyx_activity.npy",
}

SRNN_ACTIVITY_FILES = {
    "zero_bias": {
        "Lava": "03_rnn/lava_activity_noDelay_bias_zero.npy",
        "Nengo": "03_rnn/nengo_activity_noDelay_bias_zero.npy",
        "Norse": "03_rnn/norse_activity_noDelay_bias_zero.npy",
        "snnTorch": "03_rnn/snntorch_activity_noDelay_bias_zero.npy",
        "SpiNNaker2": "03_rnn/s2_activity_noDelay_bias_zero.npy",
        "Spyx": "03_rnn/spyx_activity_noDelay_bias_zero.npy",
    },
    "subtract_no_bias": {
        "Nengo": "03_rnn/nengo_activity_noDelay_noBias_subtract.npy",
        "Norse": "03_rnn/norse_activity_noDelay_noBias_subtract.npy",
        "Rockpool": "03_rnn/rockpool_activity_noDelay_noBias_subtract.npy",
        "snnTorch": "03_rnn/snntorch_activity_noDelay_noBias_subtract.npy",
        "SpiNNaker2": "03_rnn/s2_activity_noDelay_noBias_subtract.npy",
        "Spyx": "03_rnn/spyx_activity_noDelay_noBias_subtract.npy",
        "Xylo": "03_rnn/xylo_activity_noDelay_noBias_subtract.npy",
    },
}

SRNN_VARIANTS = {
    "zero_bias": {
        "nir": NIR_EXAMPLES / "braille_noDelay_bias_zero.nir",
        "metadata": NIR_EXAMPLES / "model_noDelay_bias_ref_zero.metadata.json",
        "hidden_size": 38,
        "label": "zero reset + bias",
    },
    "subtract_no_bias": {
        "nir": NIR_EXAMPLES / "braille_noDelay_noBias_subtract.nir",
        "metadata": NIR_EXAMPLES / "model_noDelay_noBias_ref_subtract.metadata.json",
        "hidden_size": 40,
        "label": "subtractive reset + no bias",
    },
}


def ensure_directories() -> None:
    for directory in (BUILD_ROOT, RUNS_ROOT, RESULTS_ROOT, CACHE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    paths = (
        REPO_ROOT / "neuro_hls/implementation_manager/implement_model.py",
        REPO_ROOT / "neuro_hls/backend/neuro_hls_functions/time_driven.h",
        REPO_ROOT / "neuro_hls/read_nir/layer_configuration.py",
        REPO_ROOT / "neuro_hls/testbench_manager/testbench_manager.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def git_state() -> Mapping[str, object]:
    def call(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return process.stdout.strip()

    return {
        "commit": call("rev-parse", "HEAD") or None,
        "dirty": bool(call("status", "--porcelain")),
    }


def vitis_version() -> Optional[str]:
    executable = shutil.which("vitis-run")
    if executable is None:
        return None
    process = subprocess.run(
        [executable, "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def provenance(paths: Mapping[str, Path], **configuration: object) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "doi": PAPER_DOI,
            "url": PAPER_URL,
            "repository": PAPER_REPOSITORY,
            "tag": PAPER_TAG,
            "commit": PAPER_COMMIT,
        },
        "neurohls": dict(git_state()),
        "vitis": vitis_version(),
        "python": sys.version,
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "configuration": configuration,
    }


def save_json(path: Path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def locate_paper_root() -> Optional[Path]:
    candidates = []
    configured = os.environ.get("NIR_PAPER_REPO")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            VALIDATION_ROOT / ".cache/NIR",
            Path("/tmp/neurohls-nir-reference"),
        ]
    )
    for candidate in candidates:
        paper = candidate if candidate.name == "paper" else candidate / "paper"
        if all((paper / name).is_dir() for name in ("01_lif", "02_cnn", "03_rnn")):
            return paper.resolve()
    return None


def reference_file(relative: str, *, allow_download: bool = True) -> Path:
    paper_root = locate_paper_root()
    if paper_root is not None and (paper_root / relative).is_file():
        return paper_root / relative
    cached = CACHE_ROOT / "paper" / relative
    if cached.is_file():
        return cached
    if not allow_download:
        raise FileNotFoundError(
            f"Official reference artefact {relative!r} is not local. Set "
            "NIR_PAPER_REPO to a checkout of neuromorphs/NIR v1.0.4."
        )
    cached.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_PAPER_BASE}/{relative}"
    partial = cached.with_suffix(cached.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(cached)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not obtain official reference artefact {relative} from {url}"
        ) from error
    return cached


def accuracy_reference(experiment: Optional[str] = None) -> pd.DataFrame:
    frame = pd.read_csv(VALIDATION_ROOT / "reference_accuracy.csv")
    if experiment is not None:
        frame = frame[frame["experiment"].str.casefold() == experiment.casefold()]
    return frame.reset_index(drop=True)


def published_similarity(name: str) -> Tuple[Sequence[str], np.ndarray]:
    payload = json.loads(
        (VALIDATION_ROOT / "reference_similarity.json").read_text(encoding="utf-8")
    )[name]
    return payload["labels"], np.asarray(payload["matrix"], dtype=np.float64)


def lif_stimulus() -> np.ndarray:
    """Build the official 1,000-step LIF stimulus from the transcribed pattern.

    The pattern is checked against the input column of a reference trace, not
    only against its step and spike counts.  A transcription slip that keeps the
    count but moves a pulse is invisible to a count check and stays invisible in
    the spike-timing metric whenever it falls after the last spike: it surfaces
    only as an inflated voltage error, which is easy to misread as numerical
    noise of the generated kernel.
    """
    stimulus = np.zeros(LIF_BASE_PATTERN.size * 10, dtype=np.int8)
    stimulus[::10] = LIF_BASE_PATTERN
    if stimulus.size != 1000 or int(stimulus.sum()) != 34:
        raise AssertionError("The official LIF stimulus must have 1000 steps and 34 spikes")

    try:
        reference = load_lif_references()["Norse"]["input"].to_numpy()
    except Exception:
        # Offline, or the reference files are unavailable: the counts above are
        # the only check that can be made.
        return stimulus
    expected = np.flatnonzero(np.asarray(reference) > 0)
    produced = np.flatnonzero(stimulus > 0)
    if not np.array_equal(expected, produced):
        divergent = sorted(set(expected.tolist()) ^ set(produced.tolist()))
        raise AssertionError(
            "LIF_BASE_PATTERN does not reproduce the official stimulus; "
            f"steps differing from the reference: {divergent}"
        )
    return stimulus


def prepare_lif_dataset() -> Path:
    ensure_directories()
    path = BUILD_ROOT / "datasets/lif_official.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    stimulus = lif_stimulus()
    if path.is_file():
        with np.load(path) as existing:
            if (
                existing["data"].shape == (1, 1000, 1)
                and np.array_equal(existing["data"].reshape(-1), stimulus)
                and np.array_equal(existing["labels"], np.asarray([0]))
            ):
                return path
    np.savez_compressed(
        path,
        data=stimulus.reshape(1, stimulus.size, 1),
        labels=np.asarray([0], dtype=np.int64),
    )
    return path


_NPZ_STREAM_CHUNK = 8 << 20


def _npz_open_member(archive: zipfile.ZipFile, member: str):
    """Open one ``.npy`` member and return its stream, shape and dtype.

    Only the array header is consumed, so the caller decides how much of the
    payload is actually decompressed.
    """
    stream = archive.open(f"{member}.npy")
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        stream.close()
        raise ValueError(f"Unsupported .npy version {version} for {member} in {archive.filename}")
    if fortran_order:
        stream.close()
        raise ValueError(f"Fortran-ordered {member} in {archive.filename} is not supported")
    return stream, tuple(shape), dtype


def npz_member_shape(path: Path, member: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Return ``(shape, dtype)`` of an ``.npz`` member without reading its data.

    ``np.load(path)[member].shape`` decompresses the whole array first, which
    for the full N-MNIST test set means materialising 14.5 GB just to read a
    dimension.  ``mmap_mode`` does not help: it is ignored for zip archives.
    """
    with zipfile.ZipFile(path) as archive:
        stream, shape, dtype = _npz_open_member(archive, member)
        stream.close()
    return shape, dtype


def _skip_bytes(stream, count: int) -> None:
    """Discard ``count`` bytes from a sequential stream."""
    while count > 0:
        chunk = stream.read(min(count, _NPZ_STREAM_CHUNK))
        if not chunk:
            raise EOFError("stream ended while skipping")
        count -= len(chunk)


def _read_bytes(stream, count: int) -> bytearray:
    """Read exactly ``count`` bytes from a sequential stream."""
    payload = bytearray(count)
    view = memoryview(payload)
    read = 0
    while read < count:
        got = stream.readinto(view[read:])
        if not got:
            raise EOFError("stream ended while reading")
        read += got
    return payload


def npz_member_rows(path: Path, member: str, start: int, stop: int) -> np.ndarray:
    """Read rows ``[start, stop)`` of an ``.npz`` member.

    The members are deflate-compressed, so there is no random access; the
    stream is decompressed sequentially but only the requested window is ever
    held in memory.  Peak usage is the size of that window instead of the size
    of the whole array.
    """
    with zipfile.ZipFile(path) as archive:
        stream, shape, dtype = _npz_open_member(archive, member)
        with stream:
            start = max(int(start), 0)
            stop = min(int(stop), shape[0])
            if stop <= start:
                return np.empty((0,) + shape[1:], dtype=dtype)
            row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            _skip_bytes(stream, start * row_bytes)
            payload = _read_bytes(stream, (stop - start) * row_bytes)
    return np.frombuffer(payload, dtype=dtype).reshape((stop - start,) + shape[1:])


def prepare_scnn_dataset(
    mode: str = "numbers", samples: int = 10, offset: int = 0,
) -> Tuple[Path, int, str]:
    """Prepare an SCNN dataset.

    ``numbers`` is the paper's ten-example activity set (one sample per digit,
    300 fixed steps). ``local`` is the fixed 1024-sample subset already present
    in this repository (300 fixed steps); it is not the paper's full
    10,000-sample accuracy protocol. ``nmnist`` is the official N-MNIST test
    set (all 10,000 samples), framed with the same 1 ms time-window binning
    the reference libraries use (see ``paper/02_cnn/sinabs_test.py`` and
    siblings), zero-padded per sample to the dataset's own max sequence
    length -- this is the like-for-like comparison set.

    ``offset`` selects a window ``[offset, offset + samples)`` of the source
    set instead of always starting at sample 0. It only applies to
    ``nmnist``, so full-dataset accuracy runs can be split into batches
    (each batch's CSim testbench data is much smaller, which avoids filling
    the disk when evaluating the full 10,000-sample test set).
    """
    ensure_directories()
    mode = mode.casefold()
    offset = max(int(offset), 0)
    if offset and mode != "nmnist":
        raise ValueError("offset batching is only supported for mode='nmnist'")
    if mode == "numbers":
        count = min(max(int(samples), 1), 10)
        scope = "paper activity set: one example per digit, 300 fixed steps"
        target = BUILD_ROOT / f"datasets/scnn_{mode}_{count}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            with np.load(target) as existing:
                if (
                    existing["data"].shape == (count, 300, 2, 34, 34)
                    and np.array_equal(existing["labels"], np.arange(count))
                ):
                    return target, count, scope
        raw = np.load(NIR_EXAMPLES / "cnn_numbers.npy", mmap_mode="r")
        if raw.shape != (300, 10, 2, 34, 34):
            raise ValueError(f"Unexpected cnn_numbers.npy shape: {raw.shape}")
        data = np.asarray(np.moveaxis(raw[:, :count], 1, 0))
        labels = np.arange(count, dtype=np.int64)
    elif mode == "local":
        count = min(max(int(samples), 1), 1024)
        scope = "local shuffled/cropped subset, 300 fixed steps; not paper full test set"
        target = BUILD_ROOT / f"datasets/scnn_{mode}_{count}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            with np.load(target) as existing:
                if existing["data"].shape == (count, 300, 2, 34, 34):
                    return target, count, scope
        source = np.load(NIR_EXAMPLES / "cnn_teste.npz")
        count = min(count, int(source["labels"].shape[0]))
        data = source["data"][:count]
        labels = source["labels"][:count]
    elif mode == "nmnist":
        source_path = NIR_EXAMPLES / "n-mnist-314-steps-10000-1ms.npz"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing full N-MNIST test-set dataset: {source_path}. Regenerate it "
                "from the local raw test set (nmnist/NMNIST/Test) with "
                "tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, "
                "time_window=1e3) -- the same framing paper/02_cnn/sinabs_test.py and the "
                "other reference-library scripts use -- zero-padded per sample to the "
                "dataset's own max sequence length."
            )
        data_shape, _ = npz_member_shape(source_path, "data")
        total = int(data_shape[0])
        steps = int(data_shape[1])
        available = max(total - offset, 0)
        if available <= 0:
            raise ValueError(f"offset {offset} is beyond the {total}-sample N-MNIST test set")
        count = min(max(int(samples), 1), available)
        scope = (
            f"official N-MNIST test set: {count} of {total} samples"
            + (f" (offset {offset})" if offset else "")
            + f", same partition and 1 ms time-window framing as the reference "
            f"libraries ({steps} zero-padded steps)"
        )
        if offset == 0 and count == total:
            return source_path, count, scope
        suffix = f"_off{offset}" if offset else ""
        target = BUILD_ROOT / f"datasets/scnn_{mode}_{count}{suffix}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if npz_member_shape(target, "data")[0] == (count, steps, 2, 34, 34):
                return target, count, scope
        data = npz_member_rows(source_path, "data", offset, offset + count)
        labels = npz_member_rows(source_path, "labels", offset, offset + count)
    else:
        raise ValueError("SCNN mode must be 'numbers', 'local', or 'nmnist'")
    np.savez_compressed(target, data=data, labels=labels)
    return target, count, scope


NMNIST_SOURCE = NIR_EXAMPLES / "n-mnist-314-steps-10000-1ms.npz"


def scnn_batch_dataset_path(count: int, offset: int) -> Path:
    """Path ``prepare_scnn_dataset`` caches an N-MNIST window at."""
    suffix = f"_off{offset}" if offset else ""
    return BUILD_ROOT / f"datasets/scnn_nmnist_{count}{suffix}.npz"


def prepare_scnn_batch_datasets(
    batch_size: int, total: Optional[int] = None, *, progress: bool = True,
) -> int:
    """Materialise every N-MNIST batch window in a single pass over the archive.

    ``prepare_scnn_dataset`` decompresses the archive up to the requested
    offset, so preparing the batches one at a time costs O(n^2) decompression
    over a full accuracy sweep.  This walks the stream once and writes each
    window as it goes; the per-batch calls then hit their on-disk cache.

    Returns the number of batch datasets that exist afterwards.
    """
    ensure_directories()
    (BUILD_ROOT / "datasets").mkdir(parents=True, exist_ok=True)
    data_shape, _ = npz_member_shape(NMNIST_SOURCE, "data")
    total = int(data_shape[0]) if total is None else min(int(total), int(data_shape[0]))
    batch_size = max(int(batch_size), 1)
    steps = int(data_shape[1])
    labels_all = npz_member_rows(NMNIST_SOURCE, "labels", 0, total)

    offsets = list(range(0, total, batch_size))
    pending = [
        offset for offset in offsets
        if not (
            scnn_batch_dataset_path(min(batch_size, total - offset), offset).is_file()
            and npz_member_shape(
                scnn_batch_dataset_path(min(batch_size, total - offset), offset), "data"
            )[0] == (min(batch_size, total - offset), steps, 2, 34, 34)
        )
    ]
    if not pending:
        if progress:
            print(f"Batch datasets already prepared: {len(offsets)} windows of up to {batch_size}")
        return len(offsets)

    pending_set = set(pending)
    written = 0
    with zipfile.ZipFile(NMNIST_SOURCE) as archive:
        stream, shape, dtype = _npz_open_member(archive, "data")
        with stream:
            row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            # Byte offset of the next unread element, so partial reads that do
            # not land on a row boundary cannot desynchronise the position.
            position = 0
            for offset in offsets:
                count = min(batch_size, total - offset)
                if offset not in pending_set:
                    continue
                _skip_bytes(stream, offset * row_bytes - position)
                position = offset * row_bytes
                payload = _read_bytes(stream, count * row_bytes)
                position += count * row_bytes
                window = np.frombuffer(payload, dtype=dtype).reshape((count,) + shape[1:])
                np.savez_compressed(
                    scnn_batch_dataset_path(count, offset),
                    data=window, labels=labels_all[offset:offset + count],
                )
                written += 1
                if progress:
                    print(f"  prepared batch offset={offset} ({written}/{len(pending)})", flush=True)
    if progress:
        print(f"Batch datasets ready: {len(offsets)} windows of up to {batch_size} ({written} new)")
    return len(offsets)


def _replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one {description} marker, found {count}; generated ABI changed"
        )
    return text.replace(old, new, 1)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _instrument_lif(project: Path) -> None:
    signature = (
        "void snn_to_hls(input_t (&input)[1], bit_t (&output)[1], "
        "bool reset_potentials)"
    )
    probe_signature = (
        "void snn_to_hls(input_t (&input)[1], bit_t (&output)[1], "
        "potential_t (&probe_voltage)[1], bool reset_potentials)"
    )
    for name in ("snn_implementation.h", "snn_implementation.cpp"):
        path = project / name
        text = _replace_once(path.read_text(encoding="utf-8"), signature, probe_signature, "LIF signature")
        if name.endswith(".cpp"):
            # The call ends in the reset-mode flag, whose value depends on the
            # metadata, so the marker is anchored on the arguments that do not
            # vary and matched up to the closing parenthesis.
            prefix = (
                "\tLIF<1,1>(layer_1, output, mem_potentials_2, tau_2, r_2, "
                "v_leak_2, v_threshold_2, v_reset_2, reset_potentials"
            )
            candidates = [
                line for line in text.splitlines() if line.startswith(prefix)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one LIF primitive call, found {len(candidates)}; "
                    "generated ABI changed"
                )
            marker = candidates[0]
            text = _replace_once(
                text, marker, marker + "\n\tprobe_voltage[0] = mem_potentials_2[0];",
                "LIF primitive",
            )
        _write_text(path, text)

    path = project / "testbench.cpp"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text, "bit_t output[OUTPUT_SIZE];",
        "bit_t output[OUTPUT_SIZE];\n            potential_t probe_voltage[1];",
        "LIF output declaration",
    )
    call = "snn_to_hls(input_data[b][s], output, s == 0);"
    traced_call = (
        "snn_to_hls(input_data[b][s], output, probe_voltage, s == 0);\n"
        "                cout << \"NEUROHLS_LIF,\" << s << \",\"\n"
        "                     << (double)input_data[b][s][0] << \",\"\n"
        "                     << setprecision(12) << (double)probe_voltage[0] << \",\"\n"
        "                     << (int)output[0] << endl;"
    )
    text = _replace_once(text, call, traced_call, "LIF testbench call")
    _write_text(path, text)


def _instrument_scnn(project: Path) -> None:
    signature = (
        "void snn_to_hls(input_t (&input)[2][34][34], bit_t (&output)[10], "
        "bool reset_potentials)"
    )
    probe_signature = (
        "void snn_to_hls(input_t (&input)[2][34][34], bit_t (&output)[10], "
        "bit_t (&probe_activity)[16][16][16], bool reset_potentials)"
    )
    for name in ("snn_implementation.h", "snn_implementation.cpp"):
        path = project / name
        text = _replace_once(path.read_text(encoding="utf-8"), signature, probe_signature, "SCNN signature")
        if name.endswith(".cpp"):
            marker = (
                "\tIF<16,16,16,1>(layer_1, layer_2, mem_potentials_2, r_2, "
                "v_threshold_2, v_reset_2, reset_potentials);"
            )
            copy = (
                marker + "\n"
                "\tfor (int c = 0; c < 16; ++c)\n"
                "\t\tfor (int y = 0; y < 16; ++y)\n"
                "\t\t\tfor (int x = 0; x < 16; ++x)\n"
                "\t\t\t\tprobe_activity[c][y][x] = layer_2[c][y][x];"
            )
            text = _replace_once(text, marker, copy, "first SCNN IF primitive")
        _write_text(path, text)

    path = project / "testbench.cpp"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text, "bit_t output[OUTPUT_SIZE];",
        "bit_t output[OUTPUT_SIZE];\n            bit_t probe_activity[16][16][16];",
        "SCNN output declaration",
    )
    text = _replace_once(
        text, "int accum_output[OUTPUT_SIZE] = {};",
        "int accum_output[OUTPUT_SIZE] = {};\n"
        "            int probe_counts[16][16][16] = {};",
        "SCNN output accumulator",
    )
    call = "snn_to_hls(input_data[b][s], output, s == 0);"
    traced_call = (
        "snn_to_hls(input_data[b][s], output, probe_activity, s == 0);\n"
        "                for (int c = 0; c < 16; ++c)\n"
        "                    for (int y = 0; y < 16; ++y)\n"
        "                        for (int x = 0; x < 16; ++x)\n"
        "                            probe_counts[c][y][x] += (int)probe_activity[c][y][x];"
    )
    text = _replace_once(text, call, traced_call, "SCNN testbench call")
    marker = "            int max_v = -1;"
    dump = (
        "            if ((cur_batch * BATCH_SIZE + b) < 10)\n"
        "            {\n"
        "                for (int c = 0; c < 16; ++c)\n"
        "                    for (int y = 0; y < 16; ++y)\n"
        "                        for (int x = 0; x < 16; ++x)\n"
        "                            cout << \"NEUROHLS_SCNN_RATE,\"\n"
        "                                 << (cur_batch * BATCH_SIZE + b) << \",\"\n"
        "                                 << c << \",\" << y << \",\" << x << \",\"\n"
        "                                 << probe_counts[c][y][x] << endl;\n"
        "            }\n\n"
        + marker
    )
    text = _replace_once(text, marker, dump, "SCNN trace dump location")
    _write_text(path, text)


def _instrument_srnn(project: Path, hidden_size: int) -> None:
    signature = (
        "void snn_to_hls(input_t (&input)[12], bit_t (&output)[7], "
        "bool reset_potentials)"
    )
    probe_signature = (
        "void snn_to_hls(input_t (&input)[12], bit_t (&output)[7], "
        f"bit_t (&probe_activity)[{hidden_size}], bool reset_potentials)"
    )
    for name in ("snn_implementation.h", "snn_implementation.cpp"):
        path = project / name
        text = _replace_once(path.read_text(encoding="utf-8"), signature, probe_signature, "SRNN signature")
        if name.endswith(".cpp"):
            pattern = re.compile(
                rf"(?P<line>\tCubaLIF<dynamics_t,1>\(layer_1, layer_3, [^\n]+\);)"
            )
            if len(pattern.findall(text)) != 1:
                raise RuntimeError("Could not locate the first generated SRNN CubaLIF")
            replacement = (
                r"\g<line>" + "\n"
                f"\tfor (int i = 0; i < {hidden_size}; ++i)\n"
                "\t\tprobe_activity[i] = layer_3[i];"
            )
            text = pattern.sub(replacement, text, count=1)
        _write_text(path, text)

    path = project / "testbench.cpp"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text, "bit_t output[OUTPUT_SIZE];",
        f"bit_t output[OUTPUT_SIZE];\n            bit_t probe_activity[{hidden_size}];",
        "SRNN output declaration",
    )
    text = _replace_once(
        text, "int accum_output[OUTPUT_SIZE] = {};",
        "int accum_output[OUTPUT_SIZE] = {};\n"
        f"            int probe_counts[{hidden_size}] = {{}};",
        "SRNN output accumulator",
    )
    call = "snn_to_hls(input_data[b][s], output, s == 0);"
    traced_call = (
        "snn_to_hls(input_data[b][s], output, probe_activity, s == 0);\n"
        f"                for (int i = 0; i < {hidden_size}; ++i)\n"
        "                    probe_counts[i] += (int)probe_activity[i];"
    )
    text = _replace_once(text, call, traced_call, "SRNN testbench call")
    marker = "            int max_v = -1;"
    dump = (
        f"            for (int i = 0; i < {hidden_size}; ++i)\n"
        "                cout << \"NEUROHLS_SRNN_RATE,\"\n"
        "                     << (cur_batch * BATCH_SIZE + b) << \",\"\n"
        "                     << i << \",\" << probe_counts[i] << endl;\n\n"
        + marker
    )
    text = _replace_once(text, marker, dump, "SRNN trace dump location")
    _write_text(path, text)


def _project_is_current(project: Path, manifest: Mapping[str, object]) -> bool:
    manifest_path = project / "validation_manifest.json"
    required = (
        "snn_implementation.cpp", "snn_implementation.h", "testbench.cpp",
        "neuron_params.h", "quantization.h", "tb_data/data.txt",
        "tb_data/targets.txt",
    )
    if not manifest_path.is_file() or not all((project / item).is_file() for item in required):
        return False
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return previous == manifest


def generate_project(
    name: str,
    nir_path: Path,
    dataset_path: Path,
    *,
    samples: int,
    steps: int,
    batch_size: int,
    metadata_path: Optional[Path] = None,
    use_float: bool = True,
    probe: Optional[str] = None,
    hidden_size: Optional[int] = None,
    force: bool = False,
) -> Path:
    """Generate a time-driven project and optionally expose CSim-only probes."""
    ensure_directories()
    nir_path = Path(nir_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    metadata_path = Path(metadata_path).resolve() if metadata_path else None
    project = BUILD_ROOT / name
    manifest = {
        "schema_version": 1,
        "probe_schema_version": PROBE_SCHEMA_VERSION,
        "name": name,
        "nir_sha256": sha256(nir_path),
        "dataset_sha256": sha256(dataset_path),
        "metadata_sha256": sha256(metadata_path) if metadata_path else None,
        "source_fingerprint": _source_fingerprint(),
        "samples": int(samples),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "backend": "time-driven",
        "parallelism_p": 0.0,
        "arithmetic": "float" if use_float else "fixed-point",
        "probe": probe,
        "hidden_size": hidden_size,
    }
    if not force and _project_is_current(project, manifest):
        return project
    if project.exists():
        shutil.rmtree(project)

    sys.path.insert(0, str(REPO_ROOT))
    from neuro_hls import NeuroHls
    from neuro_hls.backend_utils import copy_backend_to

    copy_backend_to(str(project))
    hls = NeuroHls(str(project))
    model = hls.read_nir_file(
        str(nir_path),
        metadata_file_path=str(metadata_path) if metadata_path else None,
    )
    model.define_time_driven_parallelism(0.0)
    hls.implement_model(model, use_float=use_float, backend="time-driven")
    hls.define_test_dataset(
        str(dataset_path), data_is_binary=True, step_count=int(steps),
        different_sample_per_step=True, max_samples=int(samples),
    )
    hls.create_testbench(
        total_samples=int(samples), batch_size=int(batch_size),
        reset_potentials=True, debug_mode=False,
    )

    if probe == "lif":
        _instrument_lif(project)
    elif probe == "scnn":
        _instrument_scnn(project)
    elif probe == "srnn":
        if hidden_size is None:
            raise ValueError("hidden_size is required for an SRNN probe")
        _instrument_srnn(project, int(hidden_size))
    elif probe is not None:
        raise ValueError(f"Unknown probe type: {probe}")
    save_json(project / "validation_manifest.json", manifest)
    return project


LIF_SUBTRACT_METADATA = BUILD_ROOT / "lif_subtract_metadata.json"


def lif_subtract_metadata() -> Path:
    """Write the NIR metadata that selects subtractive reset for the LIF node.

    ``lif_norse.nir`` carries a plain LIF node with no reset field, so the
    mechanism has to come from metadata.  The paper's reference trace resets by
    subtracting the threshold, and the generated kernel has to match it: with an
    assignment reset the neuron loses whatever it accumulated above threshold,
    which shifts every later spike.
    """
    ensure_directories()
    LIF_SUBTRACT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        LIF_SUBTRACT_METADATA,
        {"reset_by_subtraction": True, "reset_mechanism": "subtract"},
    )
    return LIF_SUBTRACT_METADATA


def generate_lif_project(
    *, use_float: bool = True, reset_by_subtraction: bool = False, force: bool = False
) -> Path:
    dataset = prepare_lif_dataset()
    arithmetic = "float" if use_float else "fixed"
    # The two reset modes are different experiments and must not share a build
    # directory, or the cached project of one would be reported as the other.
    suffix = "_subtract" if reset_by_subtraction else ""
    return generate_project(
        f"lif_{arithmetic}{suffix}_probe",
        NIR_EXAMPLES / "lif_norse.nir", dataset,
        metadata_path=lif_subtract_metadata() if reset_by_subtraction else None,
        samples=1, steps=1000, batch_size=1, use_float=use_float,
        probe="lif", force=force,
    )


def generate_scnn_project(
    *, mode: str = "numbers", samples: int = 10, offset: int = 0,
    use_float: bool = True, probe: bool = True, force: bool = False,
) -> Tuple[Path, int, str, Path]:
    dataset, count, scope = prepare_scnn_dataset(mode, samples, offset=offset)
    steps = int(npz_member_shape(dataset, "data")[0][1])
    project = generate_project(
        f"scnn_{mode}_{count}_{'float' if use_float else 'fixed'}"
        f"{'_probe' if probe else ''}"
        f"{f'_off{offset}' if offset else ''}",
        NIR_EXAMPLES / "cnn_sinabs.nir", dataset,
        samples=count, steps=steps, batch_size=1 if probe else min(count, 2),
        use_float=use_float, probe="scnn" if probe else None, force=force,
    )
    return project, count, scope, dataset


def generate_srnn_project(
    variant: str, *, samples: int = 140, use_float: bool = True,
    probe: bool = True, force: bool = False,
) -> Tuple[Path, int]:
    if variant not in SRNN_VARIANTS:
        raise ValueError(f"Unknown SRNN variant: {variant}")
    config = SRNN_VARIANTS[variant]
    count = min(max(int(samples), 1), 140)
    project = generate_project(
        f"srnn_{variant}_{count}_{'float' if use_float else 'fixed'}"
        f"{'_probe' if probe else ''}",
        config["nir"], NIR_EXAMPLES / "rnn_test.pt",
        metadata_path=config["metadata"], samples=count, steps=256,
        batch_size=1 if probe else min(count, 14), use_float=use_float,
        probe="srnn" if probe else None,
        hidden_size=int(config["hidden_size"]), force=force,
    )
    return project, count


def _latest_successful_run(project: Path) -> Optional[Path]:
    # ``sim.utils.slug`` currently preserves underscores.  Search by the
    # generated project name instead of duplicating that internal policy.
    candidates = sorted(
        RUNS_ROOT.glob(f"{project.name}/*/reports/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    try:
        from sim.project import hash_project
        current_hash = hash_project(project)
    except Exception:
        current_hash = None
    for summary_path in candidates:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        csim = summary.get("csim") or {}
        stages = summary.get("stages") or {}
        recorded_hash = (summary.get("project") or {}).get("project_hash")
        if current_hash is not None and recorded_hash != current_hash:
            continue
        if csim.get("available") or (stages.get("csim") or {}).get("state") == "completed":
            return summary_path.parent.parent
    return None


def run_csim(
    project: Path, *, timeout_minutes: int = 420,
    execute: bool = True, force: bool = False,
) -> Path:
    """Run through CSim with the isolated pipeline, or load the latest run."""
    project = Path(project).resolve()
    ensure_directories()
    if not execute:
        previous = _latest_successful_run(project)
        if previous is None:
            raise FileNotFoundError(
                f"No successful CSim run exists for {project.name}; enable execution"
            )
        return previous
    command = [
        sys.executable, "-m", "sim", "run", "--project", str(project),
        "--runs-dir", str(RUNS_ROOT), "--to", "csim",
        "--timeout-minutes", str(int(timeout_minutes)),
    ]
    if force:
        command.extend(["--force", "--no-reuse"])
    process = subprocess.run(
        command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=int(timeout_minutes) * 60 + 300,
        check=False,
    )
    driver_logs = RUNS_ROOT / "_driver_logs"
    driver_logs.mkdir(parents=True, exist_ok=True)
    log_path = driver_logs / f"{project.name}.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    matches = re.findall(r'"run_dir"\s*:\s*"([^"]+)"', process.stdout)
    run_dir = Path(matches[-1]).resolve() if matches else _latest_successful_run(project)
    if process.returncode != 0:
        tail = "\n".join(process.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"CSim failed for {project.name} (driver log: {log_path}):\n{tail}"
        )
    if run_dir is None:
        raise RuntimeError(f"CSim completed but its run directory was not found: {log_path}")
    return run_dir


def run_summary(run_dir: Path) -> dict:
    path = Path(run_dir) / "reports/summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def csim_log(run_dir: Path) -> Path:
    path = Path(run_dir) / "logs/csim.log"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSim log: {path}")
    return path


def csim_accuracy(run_dir: Path) -> float:
    value = (run_summary(run_dir).get("csim") or {}).get("accuracy_percent")
    if value is None:
        raise ValueError(f"The CSim log for {run_dir} has no final accuracy")
    return float(value)


def parse_lif_trace(run_dir: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"^NEUROHLS_LIF,(\d+),([-+0-9.eE]+),([-+0-9.eE]+),(\d+)\s*$"
    )
    for line in csim_log(run_dir).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            rows.append((int(match[1]), float(match[2]), float(match[3]), int(match[4])))
    frame = pd.DataFrame(rows, columns=["step", "input", "voltage", "spike"])
    if len(frame) != 1000:
        raise ValueError(f"Expected 1000 NeuroHLS LIF trace rows, found {len(frame)}")
    return frame


def parse_scnn_rates(run_dir: Path, steps: int = 300) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^NEUROHLS_SCNN_RATE,(\d+),(\d+),(\d+),(\d+),(\d+)\s*$")
    for line in csim_log(run_dir).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            rows.append(tuple(int(match[index]) for index in range(1, 6)))
    frame = pd.DataFrame(rows, columns=["sample", "channel", "y", "x", "spike_count"])
    if frame.empty:
        raise ValueError("No SCNN activity probe rows were found")
    frame["rate"] = frame["spike_count"] / float(steps)
    return frame


def parse_srnn_rates(run_dir: Path, steps: int = 256) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^NEUROHLS_SRNN_RATE,(\d+),(\d+),(\d+)\s*$")
    for line in csim_log(run_dir).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            rows.append((int(match[1]), int(match[2]), int(match[3])))
    frame = pd.DataFrame(rows, columns=["sample", "neuron", "spike_count"])
    if frame.empty:
        raise ValueError("No SRNN activity probe rows were found")
    frame["rate"] = frame["spike_count"] / float(steps)
    return frame


def load_lif_references() -> Dict[str, pd.DataFrame]:
    result = {}
    for library, relative in LIF_REFERENCE_FILES.items():
        values = np.loadtxt(reference_file(relative), delimiter=",")
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Unexpected LIF reference shape for {library}: {values.shape}")
        result[library] = pd.DataFrame(values, columns=["input", "voltage", "spike"])
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError(f"Cosine operands have different shapes: {left.shape}, {right.shape}")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def lif_comparison_table(
    references: Mapping[str, pd.DataFrame], neurohls: pd.DataFrame,
) -> pd.DataFrame:
    exact = references["Exact"]
    exact_voltage = exact["voltage"].to_numpy(dtype=float, copy=True)
    exact_voltage /= max(float(np.max(np.abs(exact_voltage))), 1e-12)
    exact_spikes = np.flatnonzero(exact["spike"].to_numpy() > 0)
    rows = []
    all_traces = dict(references)
    all_traces["NeuroHLS"] = neurohls
    for library, trace in all_traces.items():
        voltage = trace["voltage"].to_numpy(dtype=float, copy=True)
        voltage /= max(float(np.max(np.abs(voltage))), 1e-12)
        spikes = np.flatnonzero(trace["spike"].to_numpy() > 0)
        paired = min(len(exact_spikes), len(spikes))
        rows.append(
            {
                "library": library,
                "spike_count": int(len(spikes)),
                "spike_steps": " ".join(str(value) for value in spikes),
                "mean_abs_spike_lag_steps": (
                    float(np.mean(np.abs(spikes[:paired] - exact_spikes[:paired])))
                    if paired else float("nan")
                ),
                "normalised_voltage_rmse": float(np.sqrt(np.mean((voltage - exact_voltage) ** 2))),
                "normalised_voltage_cosine": cosine(voltage, exact_voltage),
            }
        )
    return pd.DataFrame(rows)


def load_scnn_reference_rates(*, allow_large_download: bool = False) -> Dict[str, np.ndarray]:
    rates = {}
    for library, relative in SCNN_ACTIVITY_FILES.items():
        path = reference_file(relative, allow_download=allow_large_download)
        activity = np.load(path, mmap_mode="r", allow_pickle=False)
        rates[library] = np.asarray(activity.mean(axis=0, dtype=np.float64)).ravel()
    return rates


def load_srnn_reference_rates(variant: str) -> Dict[str, np.ndarray]:
    if variant not in SRNN_ACTIVITY_FILES:
        raise ValueError(f"Unknown SRNN variant: {variant}")
    hidden_size = int(SRNN_VARIANTS[variant]["hidden_size"])
    rates = {}
    for library, relative in SRNN_ACTIVITY_FILES[variant].items():
        activity = np.load(reference_file(relative), allow_pickle=False)
        if activity.shape == (hidden_size, 256):
            activity = activity.T
        if activity.shape != (256, hidden_size):
            raise ValueError(f"Unexpected {library} SRNN activity shape: {activity.shape}")
        rates[library] = np.asarray(activity.mean(axis=0, dtype=np.float64)).ravel()
    return rates


def cosine_matrix(rates: Mapping[str, np.ndarray]) -> Tuple[Sequence[str], np.ndarray]:
    labels = list(rates)
    matrix = np.empty((len(labels), len(labels)), dtype=np.float64)
    for row, left in enumerate(labels):
        for column, right in enumerate(labels):
            matrix[row, column] = cosine(rates[left], rates[right])
    return labels, matrix


# Paper figure style ---------------------------------------------------------
# ``nir_paper/02_cnn/cnn_plots.ipynb`` renders the published SCNN figures with
# seaborn's "whitegrid" style, CMU Sans Serif at 20 pt, a 5x4 inch 300 dpi
# canvas and the "Blues" colormap.  seaborn is not installed in this validation
# environment, so the rcParams that "whitegrid" actually sets are applied
# directly; the result is the same look without adding a dependency.  The font
# stack degrades to DejaVu Sans on hosts without the Computer Modern faces
# instead of emitting a missing-font warning per figure.
PAPER_FONT_STACK = ("CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans")
PAPER_FIGSIZE = (5.0, 4.0)
PAPER_DPI = 300
PAPER_CMAP = "Blues"
PAPER_FONT_SIZE = 20
PAPER_ANNOTATION_SIZE = 13
# Os rótulos dos eixos herdariam os 20 pt da fonte base, o que fica maior que o
# corpo do texto quando a figura entra no documento em escala próxima de 1:1.
PAPER_TICK_SIZE = 13
# Os 30 pt vêm de cnn_plots.ipynb, onde a figura ocupava a página inteira. Aqui
# ela entra reduzida e ainda leva legenda numerada abaixo, então um título desse
# porte compete com a legenda em vez de complementá-la.
PAPER_TITLE_SIZE = 17
# O rótulo da colorbar também cairia nos 20 pt da fonte base. Fica entre os ticks
# e o título, que é a hierarquia que ele ocupa na figura.
PAPER_LABEL_SIZE = 15

_PAPER_RCPARAMS = {
    "figure.facecolor": "white",
    "figure.figsize": PAPER_FIGSIZE,
    "figure.dpi": PAPER_DPI,
    "savefig.dpi": PAPER_DPI,
    "font.family": "sans-serif",
    "font.sans-serif": list(PAPER_FONT_STACK),
    "font.size": PAPER_FONT_SIZE,
    "font.weight": "normal",
    # seaborn "whitegrid": white panel, light grey grid drawn under the data,
    # all four spines kept in the same light grey, ticks pointing outwards.
    "axes.facecolor": "white",
    "axes.edgecolor": ".8",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelcolor": ".15",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "grid.color": ".8",
    "grid.linestyle": "-",
    "grid.linewidth": 1.0,
    "text.color": ".15",
    "xtick.color": ".15",
    "ytick.color": ".15",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.bottom": True,
    "ytick.left": True,
    "image.cmap": PAPER_CMAP,
}


def apply_paper_style() -> None:
    """Match the figure style of ``nir_paper/02_cnn/cnn_plots.ipynb``."""
    mpl.rcParams.update(_PAPER_RCPARAMS)


def paper_figure(
    *, figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Return the single-panel 5x4 inch, 300 dpi canvas the paper figures use."""
    apply_paper_style()
    return plt.subplots(1, 1, figsize=figsize or PAPER_FIGSIZE, dpi=PAPER_DPI)


def paper_colorbar(
    figure: plt.Figure, mappable, axis: plt.Axes, label: str, *,
    labelpad: int = 5, decimals: int = 1,
):
    """Attach a colorbar with the paper's padding, label offset and tick format."""
    bar = figure.colorbar(mappable, ax=axis, fraction=0.046, pad=0.04)
    bar.set_label(label, labelpad=labelpad, fontsize=PAPER_LABEL_SIZE)
    bar.ax.yaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter(f"%.{decimals}f")
    )
    bar.ax.tick_params(pad=5, labelsize=PAPER_TICK_SIZE)
    return bar


def paper_palette(count: int, *, highlight: Optional[int] = None) -> list:
    """One ``Blues`` tone per bar, darkening the optional highlighted one.

    The tone is uniform on purpose: in these charts the bar position already
    carries the category, so a colour ramp across the series would encode
    nothing.  Only ``highlight`` -- the NeuroHLS bar -- earns its own value.
    """
    colormap = mpl.colormaps[PAPER_CMAP]
    if count <= 0:
        return []
    colours = [colormap(0.55)] * count
    if highlight is not None and 0 <= highlight < count:
        colours[highlight] = colormap(0.92)
    return colours


def _fitted_annotation_size(
    figure: plt.Figure, axis: plt.Axes, columns: int, widest_label: int
) -> float:
    """Largest annotation size that still fits one cell, capped at the paper's.

    ``cnn_plots.ipynb`` hardcodes 13 pt on a fixed 5x4 inch canvas, which only
    holds for the matrix size it happened to plot.  Keeping the number fixed
    here would overlap the annotations of a wider matrix, so 13 pt becomes the
    ceiling and the cell geometry sets the rest.
    """
    if columns <= 0 or widest_label <= 0:
        return PAPER_ANNOTATION_SIZE
    figure.canvas.draw()
    cell_points = (
        axis.get_window_extent().width / figure.dpi * 72.0 / columns
    )
    # A digit occupies roughly 0.6 em in these sans faces; 0.85 leaves a gutter
    # so neighbouring cells do not touch.
    return min(PAPER_ANNOTATION_SIZE, 0.85 * cell_points / (0.6 * widest_label))


def paper_title(figure: plt.Figure, axis: plt.Axes, title: str) -> None:
    """Set the title in the paper's style, shrunk to fit the fixed canvas.

    ``cnn_plots.ipynb`` hardcodes 30 pt, which fits because its titles are one
    short word ("SCNN").  The validation figures name the variant as well, so
    30 pt becomes a ceiling: a title wider than the canvas would otherwise
    stretch the saved bounding box instead of the figure keeping its 5x4 shape.
    """
    artist = axis.set_title(title, pad=10, fontsize=PAPER_TITLE_SIZE)
    figure.canvas.draw()
    available = figure.get_size_inches()[0] * figure.dpi
    used = artist.get_window_extent().width
    if used > available:
        artist.set_fontsize(PAPER_TITLE_SIZE * available / used)


def _wrapped_labels(labels: Iterable[str], width: int = 14) -> list:
    """Wrap long library names so a rotated tick does not dominate the canvas."""
    return [textwrap.fill(str(label), width) for label in labels]


def _annotation_colour(value: float, vmin: float, vmax: float) -> str:
    """Pick black or white text the way seaborn does, from cell luminance."""
    span = vmax - vmin
    normalised = 0.0 if span <= 0 else (value - vmin) / span
    red, green, blue, _ = mpl.colormaps[PAPER_CMAP](normalised)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.5 else ".15"


def save_paper_figure(figure: plt.Figure, output: Path) -> Path:
    """Write the figure to ``output`` and, alongside it, the paper's PDF copy.

    ``cnn_plots.ipynb`` saves only PDF.  The PNG is kept as well because the
    validation reports and README embed these figures by that name.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=PAPER_DPI, bbox_inches="tight")
    if output.suffix.casefold() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    return output


def plot_similarity(
    labels: Sequence[str], matrix: np.ndarray, title: str, output: Path,
) -> plt.Figure:
    """Draw the activity-similarity heatmap in the published figure's style.

    This mirrors the ``seaborn.heatmap`` call in ``cnn_plots.ipynb``: square
    cells on the ``Blues`` ramp anchored at zero, the value annotated in every
    cell, ticks mirrored on both axes and the two label sets rotated about their
    own anchors so long library names stay attached to their row and column.
    """
    figure, axis = paper_figure()
    matrix = np.abs(np.asarray(matrix, dtype=float))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap=PAPER_CMAP)
    # The heatmap carries its own cell edges; the whitegrid rules would show
    # through the colour and read as seams.
    axis.grid(False)
    axis.set_aspect("equal")

    axis.set_xticks(range(len(labels)), labels, fontsize=PAPER_TICK_SIZE)
    axis.set_yticks(range(len(labels)), labels, fontsize=PAPER_TICK_SIZE)
    annotations = [
        [f"{matrix[row, column]:.2g}" for column in range(len(labels))]
        for row in range(len(labels))
    ]
    annotation_size = _fitted_annotation_size(
        figure, axis, len(labels),
        max((len(cell) for row in annotations for cell in row), default=1),
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column, row, annotations[row][column],
                ha="center", va="center",
                color=_annotation_colour(matrix[row, column], 0.0, 1.0),
                fontsize=annotation_size,
            )

    paper_title(figure, axis, title)
    axis.xaxis.set_ticks_position("both")
    axis.yaxis.set_ticks_position("both")
    plt.setp(
        axis.get_xticklabels(),
        rotation=40, va="top", ha="right", rotation_mode="anchor",
    )
    plt.setp(
        axis.get_yticklabels(),
        rotation=45, va="bottom", ha="right", rotation_mode="anchor",
    )
    paper_colorbar(figure, image, axis, "Activity similarity")

    save_paper_figure(figure, output)
    return figure


def plot_accuracy(
    frame: pd.DataFrame, title: str, output: Path,
    *, neurohls_label: str = "NeuroHLS",
) -> plt.Figure:
    """Draw the accuracy comparison in the published figure's style.

    ``cnn_plots.ipynb`` has no bar chart -- it tabulates accuracy in LaTeX -- so
    the paper's visual language is carried over instead of copied: the same
    canvas, font and ``Blues`` ramp, with the NeuroHLS bar at the dark end of
    that ramp rather than in a second hue.
    """
    frame = frame.copy().reset_index(drop=True)
    highlight = next(
        (
            index
            for index, value in enumerate(frame["library"])
            if str(value).startswith(neurohls_label)
        ),
        None,
    )
    figure, axis = paper_figure()
    bars = axis.bar(
        _wrapped_labels(frame["library"]),
        frame["accuracy_percent"],
        color=paper_palette(len(frame), highlight=highlight),
        edgecolor=".15",
        linewidth=0.4,
    )
    axis.set_ylim(max(0.0, float(frame["accuracy_percent"].min()) - 8.0), 101.0)
    axis.set_ylabel("Accuracy (%)")
    paper_title(figure, axis, title)
    # Only the value axis is ruled, so the bars are read against the scale
    # rather than through a lattice.
    axis.grid(True, axis="y")
    axis.grid(False, axis="x")
    annotations = [f"{value:.2f}" for value in frame["accuracy_percent"]]
    annotation_size = _fitted_annotation_size(
        figure, axis, len(frame), max((len(text) for text in annotations), default=1)
    )
    for bar, value, text in zip(bars, frame["accuracy_percent"], annotations):
        axis.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=annotation_size,
        )
    plt.setp(
        axis.get_xticklabels(),
        rotation=40, va="top", ha="right", rotation_mode="anchor",
        fontsize=min(PAPER_FONT_SIZE, 90.0 / max(len(frame), 1)),
    )

    save_paper_figure(figure, output)
    return figure


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else int(default)

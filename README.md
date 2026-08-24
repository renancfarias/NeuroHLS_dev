# NeuroHLS

NeuroHLS translates a supported static NIR spiking-neural-network graph into
synthesizable C++ for FPGA high-level synthesis (HLS).  It provides two
lowerings of the same source graph:

- **time-driven** evaluates statically shaped tensors once per time step;
- **event-driven** uses scalar event streams, FIFOs, watermarks, and HLS
  `DATAFLOW` between actors.

The supported graph subset, numerical contracts, and backend restrictions are
described in [SPEC.md](SPEC.md).  Generated projects are snapshots: regenerate
them after changing the generator or runtime headers.

## Time-driven parallelism and reuse

The public time-driven parameter `p` is a normalised percentage of the
operator's static work domain, not a number of output lanes.  For a layer with
`W` statically scheduled operations, NeuroHLS derives:

```text
p = 0       -> U = 1 processing element (explicit serial sentinel)
0 < p <= 1  -> U = min(W, max(1, floor(p * W + 0.5)))
R           = ceil(W / U) reuse groups
p_eff       = U / W
I           = U * R - W idle slots in the final group
```

`Linear`/`Affine`, convolutions, and pooling count MAC or accumulation terms;
element-wise primitives count output elements; neuron primitives count neuron
updates.  The resulting `W`, `U`, `R`, `p_eff`, and `I` are written to
`parallelism_manifest.json` in every generated time-driven project.

```python
model.define_time_driven_parallelism(0.025)
model.define_time_driven_layer_parallelism("linear_1", 0.01)
```

`define_layer_parallelism(...)` remains as a deprecated alias.  There is no
separate reduction-parallelism switch: the single `p` covers the static work
domain of reduction operators too.  HLS may still report a larger initiation
interval than requested because of memory ports, accumulation dependencies,
clock constraints, or routing.

## Event-driven execution

The event-driven backend deliberately has no configurable `p`.  It retains
concurrent actors through `DATAFLOW`, bounded FIFOs, sparse operator logic,
and scalar stream transfers, but does not replicate internal actor lanes from
a global parallelism setting.  Legacy calls with `p=0` issue a deprecation
warning; non-zero event-driven settings fail explicitly.

## Validation

Run the focused parallelism tests with:

```bash
python -m unittest tests.test_time_driven_parallelism tests.test_event_driven_parallelism
```

The broader unit-test suite is:

```bash
python -m unittest discover -s tests
```

Vitis HLS synthesis and co-simulation remain necessary to establish achieved
II, resource use, timing, and power for a particular target device.

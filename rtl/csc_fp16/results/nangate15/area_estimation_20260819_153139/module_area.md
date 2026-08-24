# Module area analysis

All area values are NanGate Liberty area units. LEF cross-check is required before interpreting them as square micrometres.

## RTL hierarchy and classification

| Module | Parent | Instances/top | Input bits | Output bits | Children | Current classification | Evidence |
|---|---|---:|---:|---:|---|---|---|
| `csc_added_hw_top` | none | 1 | 1594 | 1060 | `csc_bg_engine` | BG-local wrapper | README calls the package per-BG and this module wraps one BG engine (`README.md:3-4,18,40-41`) |
| `csc_bg_engine` | `csc_added_hw_top` | 1 | 1594 | 1060 | control, partial event, batch ingress, BGA, transport | BG-local aggregate | README: one-BG structural integration (`README.md:39`) |
| `csc_descriptor_engine` | `csc_control_top` | 1 | 228 | 249 | none | BG-local | Active descriptor state for the one-BG engine (`README.md:29`) |
| `csc_agu` | `csc_control_top` | 1 | 233 | 81 | none | BG-local | Issues X/value/index requests for that engine (`README.md:30`) |
| `csc_request_tracker` | `csc_control_top` | 1 | 23 | 10 | none | BG-local/storage | Per-engine request scoreboard (`README.md:31`) |
| `csc_batch_ingress` | `csc_bg_engine` | 1 | 392 | 50 | none | BG-local/storage | Per-BG Batch8 staging (`README.md:34,18`) |
| `csc_bga` | `csc_bg_engine` | 1 | 71 | 91 | none | BG-local/storage | Explicitly one/BG (`README.md:35,18`) |
| `csc_partial_event_gen` | `csc_bg_engine` | 1 | 788 | 50 | none | BG-local/storage | Captures the BG multiply result vector (`README.md:32-33`) |
| `csc_burst_packer` | `csc_transport_top` | 1 | 69 | 261 | none | BG-local in current RTL; channel-shared candidate | Instantiated inside one-BG engine, but sharing policy is not specified; needs architecture confirmation |
| `csc_result_transport_ctrl` | `csc_transport_top` | 1 | 541 | 662 | none | BG-local in current RTL; channel/rank-shared candidate | Instantiated inside one-BG engine; controller sharing/arbitration is outside the documented architecture |

`csc_control_top`, `csc_transport_top`, and `csc_result_serializer` are intermediate hierarchy nodes. The serializer is pure wiring and has zero mapped standard cells.

## Method A: hierarchy-preserved baseline

The existing successful hierarchy-preserved run is the baseline: `results/nangate15/20260819_151723/`. Local module areas below are taken from its `stat -liberty` report.

| Preserved module | Local mapped cells | Local area | Sequential cells | Sequential area |
|---|---:|---:|---:|---:|
| `csc_descriptor_engine` | 1,342 | 623.247360 | 258 | 329.711616 |
| parameterized `csc_agu` | 527 | 156.008448 | 8 | 10.223616 |
| parameterized `csc_request_tracker` | 893 | 473.481216 | 182 | 232.587264 |
| parameterized `csc_batch_ingress` | 2,086 | 889.946112 | 393 | 502.235136 |
| parameterized `csc_bga` | 58,610 | 26,370.539520 | 9,711 | 12,410.191872 |
| `csc_partial_event_gen` | 4,173 | 1,774.977024 | 789 | 1,008.304128 |
| `csc_burst_packer` | 856 | 491.814912 | 260 | 332.267520 |
| `csc_result_transport_ctrl` | 6,981 | 3,095.642112 | 1,455 | 1,859.420160 |
| one-BG integration glue (`csc_bg_engine` local) | 1,100 | 768.638976 | 390 | 498.401280 |
| `csc_control_top` local glue | 9 | 3.293184 | 1 | 1.277952 |
| wrappers/serializer | hierarchy only | 0 | 0 | 0 |
| **top including hierarchy** | **76,577** | **34,647.588864** | **13,447** | **17,184.620544** |

The largest block is `csc_bga`: 26,370.539520 units, or 76.11% of flat top area.

## Method B: individual-top synthesis

Detailed results are in `module_area.csv`; each module was synthesized with the same `proc; opt; memory; opt; techmap; opt; dfflibmap; abc; clean; stat -liberty` sequence. Explicit top ports prevented constant-input pruning.

The sum of non-aggregate individual leaf blocks is 33,778.089984 units, 869.498880 units (2.5096%) below the reference flat top. Conversely, the explicitly re-parameterized individual full top is 34,709.372928, 61.784064 units (0.1783%) above the reference. These small differences arise because module-boundary context, re-parameterization, ABC partitioning and cross-boundary optimization change the mapped cover. The reference scaling anchor remains the audited 34,647.588864 flat result.

All mapped modules were non-empty. No `$_..._` or non-`$paramod` `$...` primitive, unmapped design cell, or inferred latch remained. The full `csc_added_hw_top` has zero constant output bits. In isolated-top checks, `csc_agu.req_addr` has five structurally zero high bits due to the configured address bases, while `csc_result_transport_ctrl` has 32 zero high bits in each of `write_req_addr` and `read_req_addr` because `RESULT_BASE=0` and its counters are 32 bits. These are partial address-width constants, not constant modules or interfaces.

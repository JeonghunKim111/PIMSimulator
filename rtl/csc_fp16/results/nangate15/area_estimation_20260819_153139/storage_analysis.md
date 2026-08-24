# Storage analysis

## Interpretation

All listed arrays are implemented as standard-cell registers and mux/compare logic in the current result. No SRAM/register-file macro area is assumed. This is therefore an **FF-based upper-bound estimate**, not a macro-based physical implementation.

Yosys issues frontend warnings converting the named arrays into register lists. Consequently the dedicated phase reports show zero retained `$mem`/`$mem_v2` objects and identical pre/post-`memory` statistics:

| Module | Pre-memory cells | Post-memory cells | Pre/post `$mem` | Post-techmap generic cells |
|---|---:|---:|---:|---:|
| `csc_bga` | 3,870 | 3,870 | 0 / 0 | 66,854 |
| `csc_request_tracker` | 255 | 255 | 0 / 0 | 1,743 |
| `csc_result_transport_ctrl` | 153 | 153 | 0 / 0 | 4,755 |

The warning is expected from asynchronous indexed reads, associative scans, many conditional entry updates and reset/valid semantics that are not directly inferred as simple single/dual-port memories.

## FF-area estimate

The mapped DFF cell is `DFFRNQ_X1` with area 1.277952. The BGA payload/tag/rank/valid arrays contain 9,664 bits, yielding a direct FF area estimate of 12,350.128128 before their muxes, comparators and decoders. The request tracker arrays/valid state contain 176 bits (224.919552 units). The realized transport arrays and their valid bits contain at least 1,060 bits (1,354.629120 units); `pending_count` is written but never read and is optimized away.

These three explicit groups account for at least 13,929.676800 units, 40.20% of the flat top. All 13,447 mapped FFs occupy 17,184.620544 units, 49.60% of flat top area. The difference includes descriptor state, batch/result staging, pointers, counters and control registers.

MUX/decoder/comparator cost is not assigned to individual arrays because synthesis shares and restructures this logic. In particular, the Q64 BGA includes 64 parallel 32-bit tag comparisons, free/hit/victim priority logic, rank update logic, and wide dynamic muxes. Its total mapped area is therefore much larger than storage FF area alone.

Per-phase `stat` reports are retained under
`module_runs/{csc_bga,csc_request_tracker,csc_result_transport_ctrl}/`.
Generated mapped netlists and verbose logs are intentionally omitted and must
be regenerated with the recorded RTL/library hashes when needed.

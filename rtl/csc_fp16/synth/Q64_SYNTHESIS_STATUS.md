# Q64 synthesis status

Primary target: `CSC_FP16_BATCH8_Q64_DEFAULT` (`BATCH8`, `IN64`, `ACC64`,
`COMPARE64`, `OUT64`).

## Completed

- Standalone Q64 BGA elaborates with Verilator.
- Full `csc_added_hw_top` elaborates with Verilator.
- Descriptor, packer, Q64 BGA and per-BG smoke simulations pass.
- The Q64 source structurally contains 64 parallel 32-bit tag equality checks,
  reduction/selection logic, 64-entry input/accumulator/output arrays and a
  6-bit bounded replacement rank per accumulator entry.

## Generic synthesis

The Q64 BGA and full `csc_added_hw_top` generic synthesis scripts elaborate in
Yosys and are exercised in GitHub Actions. These checks establish source and
generic synthesis validity, not characterized area or timing.

## Characterized area/timing

An audited Yosys 0.68+ run using the recorded NanGate 15 nm typical NLDM Liberty
mapped the full Q64 added-hardware top. The curated result reports
34,647.588864 Liberty area units/BG for the hierarchy-preserved baseline and
identifies the BGA as the dominant block. The library is not redistributed; its
SHA-256 is recorded under the result provenance directory.

This is an FF-based standard-cell mapping estimate, not placed-and-routed area.
Critical path, slack at 1 ns and Fmax remain unknown. In particular, the
simulator's one-cycle compare assumption is not treated as proof that the
64×32-bit comparison plus selection path meets a 1 ns clock.

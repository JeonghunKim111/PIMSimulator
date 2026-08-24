# BG scaling model

The general model is:

`A_CSC = A_global + N_channel * A_channel + N_BG * A_BG_local + A_storage`

Storage is already synthesized inside the module areas, so `A_storage=0` as a separate additive term in these tables; adding it again would double count.

## Scenario 1: documented per-BG implementation / upper bound

README states that the package is added per HBM bank group, that BGA replication is one/BG, and that `csc_bg_engine` is one-BG structural integration. No global/channel-shared RTL is represented. Therefore the architecture-supported model is currently identical to the requested simple upper bound:

`A_upper(N_BG) = N_BG * 34,647.588864`

At 64 BG this is **2,217,445.687296 Liberty area units**.

## Scenario 2: transport shared once (hypothesis)

If `csc_burst_packer + csc_result_transport_ctrl + serializer` is implemented once outside the BGs, its preserved hierarchy area is 3,587.457024 and the remaining per-BG area is 31,060.131840:

`A_hier(N_BG) = 3,587.457024 + N_BG * 31,060.131840`

At 64 BG this is **1,991,435.894784 units**, 226,009.792512 units (10.19%) below the simple upper bound. The current RTL has no arbitration/fan-in logic for this sharing, so this is not an implementation result and **needs architecture confirmation**.

For per-channel sharing, use `N_channel * 3,587.457024 + N_BG * 31,060.131840`; the number of channels/ranks and the missing arbitration/interconnect cost must be supplied.

## Physical-area sensitivity

`estimated_core_area = cell_area * (1 + implementation_margin) / utilization`

| Utilization | Margin | Multiplier | One-BG estimated core area |
|---:|---:|---:|---:|
| 60% | 0% | 1.666667 | 57,745.981440 |
| 60% | 10% | 1.833333 | 63,520.579584 |
| 60% | 20% | 2.000000 | 69,295.177728 |
| 65% | 0% | 1.538462 | 53,303.982868 |
| 65% | 10% | 1.692308 | 58,634.381154 |
| 65% | 20% | 1.846154 | 63,964.779441 |
| 70% | 0% | 1.428571 | 49,496.555520 |
| 70% | 10% | 1.571429 | 54,446.211072 |
| 70% | 20% | 1.714286 | 59,395.866624 |

These remain Liberty area units and heuristic utilization estimates, not place-and-route area. **LEF cross-check required.**

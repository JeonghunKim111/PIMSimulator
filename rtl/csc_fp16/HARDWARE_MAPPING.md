# Simulator-to-RTL hardware mapping

| RTL block | Simulator source/symbol | Classification | Included? |
|---|---|---|---|
| `csc_descriptor_engine` | `CSCFp16DescriptorEngine` active descriptor/chunk state | CSC-specific control | Yes |
| `csc_agu` | native `physicalAddress`, X/VALUE/INDEX_LOW/HIGH request order | CSC-specific AGU | Yes |
| `csc_request_tracker` | native `outstanding_` and typed completion match | Fixed scoreboard replacing C++ map | Yes |
| external `pim_mul_*` | `PIMBlock` FP16 MUL | Existing 256-bit PIM datapath | No |
| `csc_partial_event_gen` | descriptor engine MUL/result lane emission | CSC indexed event control | Yes |
| `csc_batch_ingress` | `Fp16BGAAdapter`, `acceptBatch` | SparsePIM/BACC-like concept, CSC adapter | Yes |
| `csc_bga` concept | `CSCFp16BankGroupAccumulator` | SparsePIM-reused BGA concept | Yes, because standalone support is measured |
| Q64 associative lookup | tag scan, `selectOldest` | Current CSC organization burden | Yes |
| external `bga_add_*` | simulator `cscFp16Add` | Existing HBM-PIM adder reuse assumption | No |
| serializer/packer | `CSCFp16PartialResultPath` 8B record and four-record burst | CSC indexed transport | Yes |
| transport controller | pending/inflight write/read state | MC control extension | Yes |
| result DRAM storage | logical resident result slots | DRAM/simulator abstraction | No; black-box interface |
| host reduction | ordered FP16 reduction/final-y | Host software | No |
| hash/oracle/loader | validation and evaluation infrastructure | Simulator/software only | No |

## Explicit exclusions

No PIM multiplier, dedicated FP16 adder, bank, row buffer, SRF/GRF/CRF, DRAM
PHY, host reduction, image loader, checksum, preflight array, debug hash,
contribution counter, dynamic container, or workload-sized resident memory is
implemented in this package.

## Boundary caveats

The result address is a linear logical burst offset from `RESULT_BASE`. This is
not a proposed physical bank/row mapping. Memory arbitration, bank conflicts,
row-buffer locality and PHY timing remain outside the package.

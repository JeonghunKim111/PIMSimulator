# PIMSimulator

## Contents

  [1. Overview](#1-overview)  
  [2. HW Description](#2-hw-description)  
  [3. Setup](#3-setup)  
  [4. Programming Guide](#4-programming-guide)  
  [5. SparsePIM Analytical Model Extension](#5-sparsepim-analytical-model-extension)
  [6. CSC M7 SpMV](#6-csc-m7-spmv)

## 1. Overview

PIMSimulator is a cycle accurate model that Single Instruction, Multiple Data (SIMD) execution units
that uses the bank-level parallelism in PIM Block to boost performance that would have otherwise used
multiple times of bandwidth from simultaneous access of all bank.
The simulator include memory and have embedded within it a PIM block, which consist of programmable
command registers, general purpose register files and execution units.

Based on https://github.com/umd-memsys/DRAMSim2, the simulator includes

* PIM Block:
  * Register files including CRF (for command), GRF (for vector value), SRF (for scalar value)
  * ALU (ADD, MUL, MAC, MAD, MOVE, FILL, NOP, JUMP, EXIT)
* PIM Kernel:
  * Generate a set of memory transactions for enabling PIM operation
* HBM2 support (refer to `ini/HBM2_samsung_2M_16B_x64.ini`)

## 2. HW description

PIM is a HBM stack that is pin compatible with HBM2 and have embedded within it a PIM block

### 2.1 Base Architecture

```C
|--------------|          |--------------|          |--------------|
|              |    (A)   |              |   (B)    |              |
|    HOST      |----------| Controller   |----------|    Memory    |
|              |          |              |          |              |
|--------------|          |--------------|          |--------------|
```

* Each channel is logically independent memory, so it has a dedicated independent controller.
* (A): Read [Addr], Write [Addr]
* (B): Activate, Read, Write, Precharge, Refresh, Activate_pim, ALU_pim, Precharge_pim, READ_pim

#### HBM2

* System Specification: system_hbm.ini
* HBM Specification: ini/HBM2_samsung_2M_16B_x64.ini
  * 1 PIM block per 2 banks, 4 Bank per Bankgroup, 4 Bank group per pseudo channel, 4 pseudo channel per die, 4 die per stack.
  * Prefetch size : 256bit
  * burst length: 4n
  * Pin speed: 2Gbps
  * The simulator supports the pseudo-channel mode only, and we assume that each pseudo-channel is totally independent.

### 2.2 Address mapping

* The address mapping is used when the memory controller decodes the address from host.
* Use Scheme8 addressing mode for PIM functionality.

```C
|<-rank->|<-row->|<-col high->|<-bg->|<-bank->|<-chan->|<-col low->|<-offset ->|
```
* the length of col_low is log(BL * JEDEC_DATA_BUS_BUTS/8), which are 5b both for HBM2
* You can also change the current addressing mode dynamically (Not recommended, though)

```C
// Static Setting in system_*.ini
ADDRESS_MAPPING_SCHEME=Scheme8
```
### 2.3 PIM Block Placement

* BANKS_PER_PIM_BLOCK = NUM_BANKS / NUM_PIM_BLOCKS

#### HBM2 case
```C
|--------|  |--------|
|        |  |        |
| BANK_0 |  | BANK_2 |
|        |  |        |
|--------|  |--------|
|  PB_0  |  |  PB_1  |
|--------|  |--------|
|        |  |        |
| BANK_1 |  | BANK_3 |
|        |  |        |
|--------|  |--------|
```
* A PIM Block (PB) is located per banks.
  * NUM_BANKS = 16, NUM_PIM_BLOCKS = 8


### 2.3 PIM Instruction-Set Architecture
|Type|Command|Description|Result (DST)|Operand (SRC0)|Operand (SRC1)|
|---|---|---|---|---|---|
|Arithmetic|ADD|addition |GRF|GRF, BANK, SRF|GRF, BANK, SRF|
|Arithmetic|MUL|multiplication |GRF|GRF, BANK|GRF, BANK, SRF|
|Arithmetic|MAC|multiply-accumulate |GRF_B|GRF, BANK|GRF, BANK, SRF|
|Arithmetic|MAD|multiply-and-add |GRF|GRF, BANK|GRF, BANK, SRF|
|Data|MOV|load or store data from register to bank|GRF, SRF|GRF, BANK||
|Data|FILL|copy data from bank to register|GRF, BANK|GRF, BANK||
|Control|NOP|do nothing||||
|Control|JUMP|jump instruction||||
|Control|EXIT|exit instruction||||

* Supports RISC-style 32-bit instructions
* Three instructions types
  * 4 Arithmetic: ADD, MUL, MAC, MAD
  * 2 Data transfer: MOV, FILL
  * 3 Control flows: NOP, JUMP, EXIT
* JUMP instruction
  * Zero-cycle static branch: supports only a pre-programmed numbers of iterations
* Operand type:
  * Vector Register (GRF_A, GRF_B)
  * Scalar Register (SRF)
  * Bank Row Buffer
* PIM instructions are stored in the Command Register File (CRF), and memory command triggers a CRF to perform a target instruction
  * each memory command increments the CRF PC
* DRAM commands decide where to retrieve data from DRAM for PIM arithmetic operations

### 2.4 Movement of Data
|Mode|Transaction|PIM Instruction|Operation|
|---|---|---|---|
|SB|Read|-|Normal Memory Read|
|SB|Write|-|Normal Memory Write|
|HAB|Write|-| PIM Write (Host to PIM Register)|
|PIM|-|MOV|read or write from bank to PIM Register|
|PIM|-|FILL|write from bank to PIM Registers|

* SB mode: standard DRAM operation
* HAB mode: Allowing concurrent accesses to multiple banks with a single DRAM command
* PIM mode: Triggers the execution of PIM instructions on the CRF by DRAM Command


## 3. Setup

### 3.1 Prerequisites
* `Scons` tool for compiling PIMSimulator:
```bash
sudo apt install scons
```
* `gtest` for running test cases:
```bash
sudo apt install libgtest-dev
```

### 3.2 Installing
* To Install PIMSimulator:
```bash
# compile
scons
```

### 3.3 Launch a Test Run
* Show a list of test cases
```bash
./sim --gtest_list_tests

# Example
PIMKernelFixture.
  gemv_tree
  gemv
  mul
  add
  relu
MemBandwidthFixture.
  hbm_read_bandwidth
  hbm_write_bandwidth
PIMBenchFixture.
  gemv
  mul
  add
  relu
```

* Test Running
```bash
# Running: functionality test (GEMV)
./sim --gtest_filter=PIMKernelFixture.gemv

# Running: functionality test (MUL)
./sim --gtest_filter=PIMKernelFixture.mul

# Running: performance test (GEMV)
./sim --gtest_filter=PIMBenchFixture.gemv

# Running: performance test (ADD)
./sim --gtest_filter=PIMBenchFixture.add
```

If you want to functionality test for other dimensions, generate a new dimension in `./data`
and add generated dimension to the source of `src/tests/KernelTestCases.cpp`.
Use the gen script in `./data` to generate data of the dimension to be changed.

### 3.4 Configuration

#### Turning on/off verbose mode
* You can select what kinds of log you want to see by modifying system_*.ini

#### Turning on/off data mode
* Data mode
  * build without -DNO_STORAGE option
* No-data mode
  * build with -DNO_STORAGE option
```bash
# build to No-data mode
scons NO_STORAGE=1
```

## 4 Programming Guide
Highly recommend you to refer to `src/tests/*` (especially, `src/tests/PIMKernel.cpp` and `src/tests/PIMBenchTestCases.cpp`)
To attach to host simulator, refer to `src/tests/PIMKernel.cpp`.
You can see commands that request memory transactions to the memory controller for GEMV or Eltwise operations on PIM.
It include a basic PIM procedure for GEMV operation in the `PIMKernel::executeGemv()`,
and also for Eltwise operation (add, mul, relu) in the `PIMKernel::executeEltwise()`

### 4.1 Primitive Function

```C
mem->addTransaction(is_read, address, tag, buffer);
```
* is_read: memory request types between READ('false') and WRITE('true')
* address: address used for memory / PIM transaction
* tag: Used for log or set to barrier. If not used, only three parameters are available,
        as `addTransaction(is_read, address, buffer)`
* buffer : used to verify pim functionality using data. Here, the buffer is at least 256-bit sized container.
If you do not want to use the data buffer, you can use it as below:
    ```C
    BurstType nullBst
    mem->addTransaction(isWrite, addr, &nullBst);
    ```

#### Memory transaction

* read
    ```C
    mem->addTransaction(false, addr, tag, buffer);
    ```
* write
    ```C
    mem->addTransaction(true, addr, tag, buffer);
    ```
Here, the buffer must be at least 256bit size container.

#### PIM transaction

* alu_pim (dataflow is similar to normal write)
    ```C
    mem->addTransaction(true, addr, tag, buffer);
    ```
  * Highly recommend you to refer to simple PIM operations using PIM ISA (`src/tests/PIMCmdGen.h`) and procedures using them(`src/tests/PIMKernel.cpp`)
  * The buffer must contain data to be broadcasted to all pim blocks of a specific channel.
  * In GEMV cacse, the corresponding weight is supplied from specific row, a specific col of multiple banks of a specific memory channel.
    * If each bank in a channel has unique ID, and bank addr in the transaction is BA, the banks satisfying (ID % BANKS_PER_PIM_BLOCK == BA) supply the weight to PIM blocks.
      * if BANKS_PER_PIM_BLOCK == 2, and BA = 0, bank 0,2,4,6,... supply the weight to pim-block 0,1,2,3,..., respectively.
      * if BANKS_PER_PIM_BLOCK == 2, and BA = 1, bank 1,3,5,7,... supply the weight to pim-block 0,1,2,3,..., respectively.
    * As a result, data broadcasted and data from multiple banks of a specific channel are multiplied and accumulated.

* read_pim (dataflow is similar to normal read)
    ```C
    mem->addTransaction(false, addr, tag, buffer);
    ```
   * Read the accumulated partial sum in the pim block. Then reset the buffer.
      * If each pim block in a channel has unique ID, and bank addr in the transaction is BA, the PIM block satisfying (ID == BA) supply partial sums to DQ.
   * Highly recommend to use the address that alu_pim command used at the last

### 4.2 PIM High-level Steps
* The following shows the high level steps of a generic PIM operation.
   * Place data in DRAM
   * Switch to HAB mode
   * Program CRF
   * Enable PIM
   * Execute PIM
   * Disable PIM
   * Switch to SB mode

* A similar procedure at the source level can be found in `src/tests/PIMKernel.cpp`.
```C
    /* Example Code - PIMKernel::executeELtwise() */

    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);      // Switch to HAB
    programCrf(pim_cmds);                            // Program CRF
    changePIMMode(dramMode::HAB, dramMode::HAB_PIM); // Enable PIM

    if (ktype == KernelType::ADD || ktype == KernelType::MUL)
        computeAddOrMul(num_tile, input0_row, result_row, input1_row); // Execute PIM
    else if (ktype == KernelType::RELU)
        computeRelu(num_tile, input0_row, result_row);

    changePIMMode(dramMode::HAB_PIM, dramMode::HAB); // Disable PIM
    changePIMMode(dramMode::HAB, dramMode::SB);      // Switch to SB mode
    parkOut();

```

* The other basic operation flow on PIM for GEMV(Matrix Vector multiplication), Element-wise operation are described in the `src/tests/PIMKernel.cpp`.

## 5. SparsePIM Analytical Model Extension

This fork also contains an in-progress analytical model for SparsePIM-style SpMV.
The goal is to reproduce the SparsePIM paper's GPU-baseline speedups for the
paper sparse matrix workloads, then use the validated model to evaluate
algorithmic variants of SparsePIM preprocessing and execution.

The SparsePIM preprocessing inputs are expected under:

```bash
../SparsePIM/guided_kmeans_coo_results/<matrix>/
```

Each matrix directory should contain the Guided K-means COO preprocessing outputs
used by the model:

```text
reordered_matrix.txt
column_permutation.txt
clusters.txt
```

The paper workload mapping and target speedups are recorded in:

```text
SPARSEPIM_WORKLOAD_TARGETS.md
```

### 5.1 Latency Scope

The target comparison scope follows the SparsePIM paper:

* GPU baseline: host-to-device input transfer, cuSPARSE SpMV kernel execution,
  and device-to-host result retrieval per iteration.
* SparsePIM: kernel programming/init, PIM computation, and result retrieval per
  iteration.

The model therefore reports end-to-end per-iteration latency estimates, not only
the PIM compute phase.

### 5.2 Model Variants

The original DRAF+BGA model variants are preserved:

```bash
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_model
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_conservative_model
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v2_model
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v2_conservative_model
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v21_model
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v21_conservative_model
```

Two newer structural variants were added for DRAF padding analysis:

```bash
# v3: charges all DRAF padded NZE slots as exposed work
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v3_structural_model

# v4: charges only critical-path DRAF padding at the bank-group level
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v4_structural_model

# v5: starts from v4 critical-path padding and exposes hidden padding
# according to fragmentation, memory expansion, and BG imbalance
./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v5_structural_model
```

To run a single matrix:

```bash
SPMV_BENCH_MATRIX=Stanford ./sim --gtest_filter=ClusteredSpmvBenchFixture.sparsepim_guided_kmeans_coo_draf_bga_v4_structural_model
```

### 5.2.1 CSC outer-product partial-stream baseline

`CSCPartialStreamSpMV.cpp` provides a baseline that does not use DRAF, BGA,
clustering, or PIM-side indexed accumulation.  It preserves CSC column order,
maps column `c` to `c % global_bank_groups`, generates one partial value per
NNZ, reads only those values back, and performs deterministic host scatter
reduction with the host-resident CSC row indices.

Build and run the functional corner cases with:

```bash
scons -j4
./sim --gtest_filter=CSCPartialStreamFunctionalTest.*
```

Run a Matrix Market or zero-based CSC-ordered triplet matrix with:

```bash
CSC_PARTIAL_MATRIX=../SparsePIM/csc_partial_stream_sample.mtx \
CSC_PARTIAL_OUTPUT=../SparsePIM/csc_partial_stream_results.csv \
CSC_PARTIAL_ITERATIONS=7 \
./sim --gtest_filter=CSCPartialStreamBenchmark.RunFromEnvironment
```

The functional path uses FP32 on the CPU to validate partial products and host
reduction.  The timing path uses the simulator's configured channel/rank/BG
topology, existing MUL CRF encoding, and DRAM transactions for matrix/x
placement, value-stream triggers, FP32-equivalent partial writes, and partial
readback.  The public dense `executeEltwise()` API cannot schedule independent
variable-length columns or safely reload SRF per BG, so column scalar loads and
MUL triggers are represented as an SB-mode memory timing trace after the real
MUL CRF is programmed.  Arithmetic values and ALU latency are therefore not
functionally simulated by PIMSimulator; this boundary is intentional and is
reported as a trace-based baseline rather than a PIM-only CSC kernel.


### 5.2.2 Trace-based direct CSR inner-product baseline

`CSRDirectSpMV.cpp` is an independent minimal-placement CSR baseline.  It
preserves the stored row order and the entry order inside every row, assigns
row `r` to global bank group `r % global_bank_groups`, and never sorts,
clusters, permutes, splits, or NNZ-balances rows.  Empty rows remain explicit
output rows.  The current configuration models one HBM2 stack: 4 bank groups per
pseudo-channel, 4 pseudo-channels per die, and 4 dies per stack. Because the
simulator represents each pseudo-channel as an independent channel, this is 16
channels, 1 rank per channel, and 4 bank groups per rank, for 64 global bank
groups. Representative addresses generated by `PIMAddrManager::addrGenSafe()`
are decoded and checked against that physical channel/rank/BG tuple.

The input files are standard SciPy CSR NPZ archives.  The C++ build has no ZIP
dependency, so `tools/export_csr_npz.py` strictly checks `format == "csr"` and
serializes `shape`, `indptr`, `indices`, and `data` to a temporary binary file.
It does not call `sort_indices`, `sum_duplicates`, or a sparse-format
conversion.  FP64 values are converted to the logical FP32 timing/functional
payload during input loading; that time is reported as `input_load_us`, not as
kernel time.  Run a single matrix with:

```bash
CSR_DIRECT_MATRIX=../SparsePIM/sparse_matrix_csr/cant_csr.npz \
CSR_DIRECT_OUTPUT=../SparsePIM/csr_direct_round_robin_results.csv \
CSR_DIRECT_ITERATIONS=7 \
./sim --gtest_filter=CSRDirectSpMVBenchmark.RunFromEnvironment
```

Run all files ending in `_csr.npz`, in sorted filename order, with:

```bash
CSR_DIRECT_MATRIX_DIR=../SparsePIM/sparse_matrix_csr \
CSR_DIRECT_OUTPUT=../SparsePIM/csr_direct_all_results.csv \
CSR_DIRECT_ITERATIONS=7 \
./sim --gtest_filter=CSRDirectSpMVBenchmark.RunDirectory
```

The functional path computes FP32 CSR SpMV twice (the original CSR reference
and the BG-sharded row-local path) and compares every output.  The timing path
programs the existing MAC CRF and emits PIMSimulator memory transactions for
matrix placement, one non-replicated x placement, row descriptors, packed
values, packed column indices, one physical `x[col_idx]` gather per NNZ, one
FP32 output write per row, and full-y readback.  Because the public PIM API has
no indexed gather or variable-length per-BG row MAC interface, the CSR body is
issued in SB mode as a host-generated memory timing trace.  The simulator does
not functionally execute the FP32 accumulator or add native MAC-pipeline
latency.  Consequently this is a **CSR functional validation with PIM
memory-timing trace**, not cycle-accurate native CSR PIM execution.

Every reported timing phase is drained before the next phase, so phase overlap
and pipelining are intentionally excluded.  `csr_execution_trace_cycle`
contains descriptor/value/index/x-gather traffic and is not also counted as
separate sub-phases.  The key resident metrics are
`modeled_core_kernel_cycle = crf + execution_trace + output_write` and
`resident_iteration_cycle = x_transfer + mode_change + modeled_core_kernel +
output_readback`.  Matrix placement occurs once; resident iterations include
one unreported warm-up and the requested median of at least the configured
iteration count.

Run the functional, address-mapping, and trace sanity tests with:

```bash
./sim --gtest_filter='CSRDirectSpMVFunctionalTest.*:CSRDirectSpMVPhysicalMappingTest.*:CSRDirectSpMVSanityTest.*'
```

### 5.3 Current Findings
The v3 model was introduced after observing that v2.1 significantly
overestimated SparsePIM speedup for highly sparse graph/web workloads such as
Stanford and webbase-1M. The v3 model adds an explicit DRAF padded-zero work
term:

```text
padded_zero_compute_cycle = ceil(draf_nze_padding / 16)
```

This matches the SparsePIM paper's explanation that DRAF padding increases
memory usage and operation count under synchronous column access. It brought
Stanford and webbase-1M close to their target speedups, but it over-penalized
many regular or denser workloads.

The v4 model then replaced total padding with a bank-group critical-path padding
estimate:

```text
actual_group_step = ceil(DRAF column groups in a logical bank group / 2 banks)
ideal_group_step  = (nonzeros in the same logical bank group / 16 NZEs) / 2 banks
critical_padding  = sum(max(0, actual_group_step - ideal_group_step))
```

This improved the overall geometric mean compared with v3, but it currently
overestimates speedup for fragmented sparse workloads such as
soc-sign-epinions, Stanford, and webbase-1M. The current interpretation is that
critical-path padding alone hides too much of the DRAF format conversion,
memory-expansion, and synchronous dummy-access overhead for low-NNZ-column
workloads.

The next structural refinement should combine critical-path padding with an
exposure term derived from:

```text
single_nnz_column_ratio
low_nnz_column_ratio
draf_memory_expansion
bg_imbalance
```

The detailed model notes are in:

```text
SPARSEPIM_V3_STRUCTURAL_MODEL.md
```

Both v3 and v4 print machine-readable result markers:

```text
V3_RESULT_CSV,<matrix>,<gpu_ms>,<target_speedup>,<target_pim_ms>,<model_ms>,<model_speedup>,...
V4_RESULT_CSV,<matrix>,<gpu_ms>,<target_speedup>,<target_pim_ms>,<model_ms>,<model_speedup>,...
V5_RESULT_CSV,<matrix>,<gpu_ms>,<target_speedup>,<target_pim_ms>,<model_ms>,<model_speedup>,...
V6_RESULT_CSV,<matrix>,<gpu_ms>,<target_speedup>,<target_pim_ms>,<model_ms>,<model_speedup>,...
```

When the structural model tests are run for all 16 workloads, they also write
summary files:

```text
spmv_guided_kmeans_draf_bga_v3_structural_results.txt
spmv_guided_kmeans_draf_bga_v4_structural_results.txt
spmv_guided_kmeans_draf_bga_v5_structural_results.txt
spmv_guided_kmeans_draf_bga_v6_structural_results.txt
spmv_guided_kmeans_draf_bga_v3_phase_diagnostics.txt
spmv_guided_kmeans_draf_bga_v4_phase_diagnostics.txt
spmv_guided_kmeans_draf_bga_v5_phase_diagnostics.txt
spmv_guided_kmeans_draf_bga_v6_phase_diagnostics.txt
```

The v6 structural model keeps v5 padding exposure and separates BGA accumulation
into raw, hidden, and exposed cycles. Its first full-suite run improves GMean
model speedup from v5 `1.477x` to v6 `1.809x`, while the rounded paper-target
GMean is `2.197x`.

## 6. CSC M7 SpMV

The M7 CSC path models descriptor-driven PIM multiplication, bank-group-local
associative accumulation, bounded partial-result writeback, host readback, and
deterministic indexed FP32 reduction into the final output vector.

The repository includes the image preprocessor needed to run this path from a
sparse matrix; a separate SparsePIM checkout is not required. Input matrices
use the text CSC format described in
[`tools/csc_light_preprocess/README.md`](tools/csc_light_preprocess/README.md).

First build PIMSimulator:

```bash
scons
```

Generate a physical image from the included toy matrix:

```bash
python3 tools/csc_light_preprocess/csc_light_preprocess.py \
  --matrix tools/csc_light_preprocess/testdata/toy_csc.txt \
  --layout csc_aligned --policy round_robin --segment-nnz 0 \
  --warmup 0 --repeat 1 --no-csv \
  --export-image /tmp/csc_toy_image
```

Run the observable M7 end-to-end test:

```bash
CSC_EXTERNAL_IMAGE=/tmp/csc_toy_image \
  ./sim --gtest_filter=CSCM7BFullRunTest.ExternalToyPrintsFullCycleBreakdown
```

The export directory must be absent or empty because the preprocessor does not
overwrite an existing image. Replace the toy `--matrix` path with another
matrix in the same format for other workloads. See
[`docs/csc/CSC_IMAGE_FORMAT.md`](docs/csc/CSC_IMAGE_FORMAT.md) for the binary
contract and [`docs/csc/M7B_IMPLEMENTATION_AND_TOY_RESULTS.txt`](docs/csc/M7B_IMPLEMENTATION_AND_TOY_RESULTS.txt)
for the implemented pipeline and recorded toy result.

### Contact
* Shin-haeng Kang (s-h.kang@samsung.com)
* Sanghoon Cha (s.h.cha@samsung.com)
* Seungwoo Seo (sgwoo.seo@samsung.com)
* Jin-seong kim (jseong82.kim@samsung.com)

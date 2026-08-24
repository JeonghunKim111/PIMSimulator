# PIMSimulator

## Final CSC SpMV releases

This branch is the final FP16 implementation of the CSC outer-product SpMV
pipeline and is the latest production branch.

- FP16: branch `fp16-final`; historical M7 baseline tag `fp16-m7-final`
- FP32: branch `fp32-final`, tag `fp32-m7-final`
- FP16 architecture: 16-lane binary16 compute, BG-decoupled descriptor
  execution, BG-local BGA, partial-result writeback/readback, and deterministic
  indexed reduction

Build and run the core FP16 regression suite:

```bash
scons
./sim --gtest_filter='CSCFp16*.*:FP16SemanticsCharacterizationTest.*:CSCFP32GoldenBaselineTest.*'
```

External matrix tests are opt-in and require a verified FP16 CSC v2 image as
documented under `docs/csc/`.

## Recovery from a clean machine

This fork is derived from Samsung SAIT's public PIMSimulator repository:

- upstream repository: `https://github.com/SAITPublic/PIMSimulator.git`
- pinned upstream `dev` commit: `3703d1f19c8f027360cc33a3243eb271e3bb6898`
- simulator branch: `fp16-final`

Restore and verify the source relationship with:

```bash
git clone https://github.com/JeonghunKim111/PIMSimulator.git
cd PIMSimulator
git switch fp16-final
git remote add samsung https://github.com/SAITPublic/PIMSimulator.git
git fetch samsung dev
git merge-base --is-ancestor 3703d1f19c8f027360cc33a3243eb271e3bb6898 HEAD
```

On Ubuntu 24.04, install the dependencies and run the self-contained FP16
regression with:

```bash
sudo apt update
sudo apt install build-essential scons libgtest-dev
scons -j4
./sim --gtest_filter='CSCFp16*.*:FP16SemanticsCharacterizationTest.*:CSCFP32GoldenBaselineTest.*'
```

The FP16 M7 simulator exposes four measurement scopes:

- `COMPUTE_ONLY`: operand timing, 16-lane multiplication, and partial capture;
- `BGA_VALIDATION`: BGA plus untimed direct replay for functional validation;
- `TRANSPORT_ONLY`: BGA, writeback, and readback without host reduction; and
- `END_TO_END_TIMED`: BGA, writeback, readback, and host reduction.

Q64 is the default architectural evaluation preset. Set
`CSC_FP16_BGA_CAPACITY=16` for the Q16 sensitivity run. This environment
variable scales the input queue, accumulator, compare width, and output queue
together, so these presets are not claims of structurally faithful SparsePIM
RTL equivalence.

Large matrix images are intentionally excluded. Clone
`https://github.com/JeonghunKim111/SparsePIM.git` as a sibling directory and
follow its data-reproduction procedure before running the opt-in external image
tests.

## Contents

- [1. Overview](#1-overview)
- [2. HW Description](#2-hw-description)
- [3. Setup](#3-setup)
- [4. Programming Guide](#4-programming-guide)
- [5. SparsePIM Analytical Model Extension](#5-sparsepim-analytical-model-extension)
- [6. Running CSC M7 SpMV](#6-running-csc-m7-spmv)
- [7. CSC M7 Dataflow and Data Format](#7-csc-m7-dataflow-and-data-format)
- [8. Hardware Added for CSC SpMV](#8-hardware-added-for-csc-spmv)

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

## 6. Running CSC M7 SpMV

The M7 CSC path models descriptor-driven PIM multiplication, bank-group-local
associative accumulation, bounded partial-result writeback, host readback, and
deterministic indexed FP32 reduction into the final output vector.

The repository includes the image preprocessor needed to run this path from a
sparse matrix; a separate SparsePIM checkout is not required. Input matrices
use the text CSC format described in
[`tools/csc_light_preprocess/README.md`](tools/csc_light_preprocess/README.md).

### 6.1 Requirements and build

Install SCons, GoogleTest, Python 3 with NumPy, and a C++17 compiler with
OpenMP support. On Ubuntu, build the simulator from the repository root with:

```bash
scons
```

The preprocessor's C++ helper is compiled automatically on first use. It can
also be built explicitly with `make -C tools/csc_light_preprocess`.

### 6.2 Input matrix

The preprocessor accepts a headerless sparse triplet file in CSC column order.
Each line contains:

```text
<zero-based row> <zero-based column> <floating-point value>
```

Column indices must be nondecreasing; row order inside a column is preserved.
The parser infers dimensions from the largest indices and constructs
conventional `col_ptr`, `row_idx`, and FP64 `values` arrays internally. Inputs
whose dimensions depend on trailing completely empty rows or columns need an
explicit nonzero representation or conversion before using this loader. See
[`tools/csc_light_preprocess/testdata/toy_csc.txt`](tools/csc_light_preprocess/testdata/toy_csc.txt)
for a complete example.

### 6.3 Generate an HBM-PIM physical image

Generate a physical image from the included toy matrix:

```bash
python3 tools/csc_light_preprocess/csc_light_preprocess.py \
  --matrix tools/csc_light_preprocess/testdata/toy_csc.txt \
  --layout csc_aligned --policy round_robin --segment-nnz 0 \
  --warmup 0 --repeat 1 --no-csv \
  --export-image /tmp/csc_toy_image
```

The M7 format requires `--layout csc_aligned`, 64 bank groups, and
`--segment-nnz 0`. `round_robin` deterministically assigns nonempty columns to
BGs. The output directory must be absent or empty because image export never
overwrites existing data. A successful run prints JSON with
`"verification_passed": true` and creates `manifest.json` plus four files for
each of the 64 BGs.

For another matrix, replace the `--matrix` and `--export-image` paths. Mapping
policies `load_only` and `load_similarity` are also available for experiments;
start with `round_robin` when validating a new input.

### 6.4 Run the HBM-PIM CSC SpMV simulation

Run the observable toy end-to-end test:

```bash
CSC_EXTERNAL_IMAGE=/tmp/csc_toy_image \
  ./sim --gtest_filter=CSCM7BFullRunTest.ExternalToyPrintsFullCycleBreakdown
```

For a non-toy matrix, use the general full-run test:

```bash
CSC_EXTERNAL_IMAGE=/tmp/my_csc_image \
  ./sim --gtest_filter=CSCM7BFullRunTest.ExternalMatrixPrintsFullCycleBreakdown
```

The test constructs a deterministic FP32 input vector `x`, runs the complete
descriptor/PIM/BGA/writeback/readback/reduction path, and compares the final
vector with a CPU FP32 reference. Successful output must report:

```text
contribution_conservation: PASS
record_conservation: PASS
byte_conservation: PASS
resultValid: PASS
endToEndSpMVComplete: PASS
CPU_reference_comparison: PASS
```

It also prints compute, BGA drain, writeback, readback, host reduction, and
end-to-end cycle counts. Large matrices can require substantial simulation
time and resident partial-result capacity, so validate the toy image first and
then increase workload size gradually.

### 6.5 Reproduction checklist

1. Build `sim` with `scons`.
2. Convert the matrix to the required text CSC form if necessary.
3. Export a `csc_aligned` physical image with the bundled preprocessor.
4. Confirm that preprocessing reports `verification_passed: true`.
5. Set `CSC_EXTERNAL_IMAGE` to the exported image directory.
6. Run the toy or general M7B full-run GoogleTest.
7. Check final-result correctness and all three conservation chains.

## 7. CSC M7 Dataflow and Data Format

### 7.1 End-to-end dataflow

```text
Host text CSC matrix
        |
        v
CSC image preprocessor
  column-to-BG mapping, FP64-to-FP32 conversion,
  32-byte alignment, descriptor and x-slot generation
        |
        v
64 authoritative BG-local physical images
        |
        v
CSC descriptor engine
  load x scalar -> read value/index chunk -> masked FP32 MUL
        |
        v
One associative BGA per global bank group
  indexed accumulation by row -> eviction/final-drain partials
        |
        v
8-byte partial records: { uint32 row_idx, float32 value }
        |
        v
Per-BG 32-byte burst packer (4 records per full burst)
        |
        v
Bounded partial-result writeback -> resident burst buffers
        |
        v
POST_WRITEBACK host readback -> bounded return queue
        |
        v
Deterministic indexed FP32 host reduction
        |
        v
Authoritative final y / resultValid()
```

Each nonempty input column is owned by exactly one global BG. Its descriptor
identifies the BG-local value and row-index streams and the corresponding
input-vector `x` slot. A descriptor is processed in chunks of eight NNZs, the
native SIMD width. Tail chunks mask inactive lanes, so padding never executes
a multiply or creates a partial result.

Products are accumulated by row in the owning BG's bounded associative Bank
Group Accumulator (BGA). BGA capacity eviction and final drain emit physical
partial records. Backpressure is propagated through bounded queues rather than
using an unbounded result container.

M7 uses `POST_WRITEBACK` readback: host reads begin only after all BGA outputs
have been packed, written, and made resident. Returned records are reduced in a
stable host-visible order using an FP32 addition for every record. The host
reduction, rather than a validation shadow or CPU reference, is the
authoritative final output.

### 7.2 Input and logical CSC representation

The input is the column-ordered `row column value` triplet form described in
Section 6.2. During preprocessing it becomes conventional compressed sparse
column arrays:

| Array | Type | Entries | Meaning |
|---|---|---:|---|
| `col_ptr` | integer | `num_columns + 1` | Start/end positions of each column |
| `row_idx` | integer | `nnz` | Zero-based output row for each nonzero |
| `values` | floating point | `nnz` | Nonzero matrix values |

For column `c`, entries occupy `[col_ptr[c], col_ptr[c + 1])`. SpMV computes
`y[row_idx[k]] += values[k] * x[c]` for each entry in that interval.

### 7.3 Physical BG-partitioned image

The preprocessor converts the logical CSC arrays into a directory containing:

```text
manifest.json
bg_00_values.bin
bg_00_row_idx.bin
bg_00_descriptors.bin
bg_00_x_permutation.bin
...
bg_63_values.bin
bg_63_row_idx.bin
bg_63_descriptors.bin
bg_63_x_permutation.bin
```

Version 1 of the image contract uses little-endian encoding, 64 global BGs,
eight FP32 SIMD lanes, 32-byte bursts/alignment, FP32 values, and `uint32` row
indices. Each BG has independent value and row-index streams. Every nonempty
column begins on a 32-byte boundary and is padded to the next boundary. Padding
is zero-filled physical storage and is not counted as an NNZ.

Each descriptor is exactly 32 bytes:

| Byte offset | Size | Field |
|---:|---:|---|
| 0 | 8 | BG-local value-stream byte offset |
| 8 | 8 | BG-local row-index-stream byte offset |
| 16 | 4 | Column NNZ count |
| 20 | 4 | BG-local `x` slot |
| 24 | 4 | Original matrix column |
| 28 | 4 | Global BG ID |

The offsets are BG-local stream offsets, not absolute HBM addresses.
PIMSimulator combines the global BG ID and local offset with the configured
Scheme8 channel/rank/BG/bank topology to generate physical requests. Values,
row indices, and packed `x` use separate bank/base-row regions; M7 partial
results then follow the modeled writeback/readback path described above.

`manifest.json` records dimensions, mapping policy, per-BG sizes and counts,
column ownership, conversion statistics, and FNV-1a checksums. The simulator
validates all sizes, checksums, alignment, descriptor ownership, x-slot
permutations, row ranges, and total NNZ before execution.

### 7.4 Partial-result and completion contracts

A BGA output is serialized as one 8-byte architectural record:

| Field | Type | Bytes |
|---|---|---:|
| `row_idx` | `uint32` | 4 |
| `value` | IEEE-754 `float32` | 4 |

Four records form a full 32-byte writeback burst. A BG's final one to three
records form one zero-padded 32-byte tail burst. Simulator-only ownership and
sequence metadata is not counted as transferred payload.

Production completion requires compute submission, BGA drain, partial
writeback, host readback, host reduction, and all conservation checks to finish
without a sticky error. The expected invariants are:

```text
matrix NNZ = accepted contributions = reduced contributions
BGA records = packed records = readback records = reduced records
write useful + padding bytes = write transferred bytes
read useful + padding bytes  = read transferred bytes
```

The complete binary contract is in
[`docs/csc/CSC_IMAGE_FORMAT.md`](docs/csc/CSC_IMAGE_FORMAT.md). Implementation
details and the recorded toy timeline are in
[`docs/csc/M7B_IMPLEMENTATION_AND_TOY_RESULTS.txt`](docs/csc/M7B_IMPLEMENTATION_AND_TOY_RESULTS.txt).

## 8. Hardware Added for CSC SpMV

### 8.1 Design objective and baseline topology

The original PIMSimulator provides HBM2 channels, ranks, banks, and programmable
SIMD PIM blocks, but its legacy PIM commands operate at rank-wide granularity
and do not provide indexed sparse accumulation. The CSC extension preserves
that datapath and adds control, accumulation, and result-transport structures
needed to execute independently partitioned sparse columns.

The production CSC configuration has 16 channels, one rank per channel, four
bank groups per rank, 16 banks per rank, and eight PIM blocks per rank. Two
physical PIM blocks belong to each local BG:

| Local BG | PIM blocks | Banks |
|---:|---|---|
| 0 | 0, 1 | 0-3 |
| 1 | 2, 3 | 4-7 |
| 2 | 4, 5 | 8-11 |
| 3 | 6, 7 | 12-15 |

Thus the complete simulated system exposes 64 global BG execution domains.
Global BG `g` maps to channel `g / 4`, rank 0, and local BG `g % 4`. The
topology is checked at construction; unsupported bank/PIM-block organizations
are rejected rather than silently remapped.

### 8.2 Added hardware-model components

| Component | Placement/scope | Function |
|---|---|---|
| CSC descriptor engine | One logical worker per global BG | Sequences descriptor fetch, `x`, value and index reads, chunks columns into eight lanes, and tracks request completion |
| BG-targeted operation interface | Descriptor engine to `PIMRank` | Carries destination BG, PIM-block mask, operands, valid-lane count, sequence and generation identity |
| Rank-local targeted arbiter | One per rank | Grants at most one ready BG operation per rank per cycle using deterministic round robin and shared-resource checks |
| Masked SIMD execution | Existing physical `PIMBlock` datapath | Executes FP32 multiplication only for the valid lanes of full or tail chunks |
| Bank Group Accumulator (BGA) | One independent instance per global BG | Associatively merges products with equal row indices and emits capacity-eviction or final-drain partials |
| Stable BGA output port | One per BGA | Holds the front output until a bounded downstream destination reserves and accepts it |
| Partial-result packer | One per global BG | Packs four 8-byte indexed partials into each 32-byte writeback burst and creates final tail bursts |
| Writeback transaction model | Rank-arbitrated, BG-resident storage | Models bounded pending/in-flight writes, latency, buffer reservation, and resident burst ownership |
| Host readback model | Channel-arbitrated | Models bounded read issue, read latency, in-flight limits, and a bounded host return queue |
| Host FP32 reduction model | Host side | Deterministically accumulates returned indexed records into the authoritative final vector |

### 8.3 Descriptor-driven sparse execution control

`CSCDescriptorEngine` is the new sparse-operation controller. For each owned
column it loads one scalar `x[col]`, then requests aligned value and row-index
bursts. Each 32-byte chunk supplies up to eight FP32 values and eight `uint32`
row indices. The engine creates a `MASKED_MUL` operation only after both
operands are ready.

The controller tracks each memory request and targeted operation with explicit
owner, kind, sequence, and generation fields. It rejects unknown, stale,
duplicate, or incorrectly routed completions. Different BG engines advance
independently under the `BG_DECOUPLED` scheduling policy; a stalled BG does not
form a global descriptor barrier.

The primary state sequence is:

```text
FETCH_DESCRIPTOR -> LOAD_X -> FETCH_VALUE/FETCH_ROW_INDEX
 -> WAIT_OPERANDS -> SIMD_MUL
 -> WAIT_TARGET_GRANT -> WAIT_TARGET_COMPLETION
 -> EMIT_PARTIALS -> ADVANCE_CHUNK/NEXT_DESCRIPTOR
```

### 8.4 BG-targeted PIM execution

The extension adds a BG-targeted execution mode to `PIMRank`. A targeted
operation identifies the channel, rank, local BG, selected PIM block, opcode,
valid lane count, and exactly-once identity. This avoids broadcasting every CSC
chunk to all PIM blocks in a rank.

Each rank owns one depth-one pending slot and one active-operation slot per
local BG. A round-robin arbiter scans ready BGs and grants at most one operation
per rank per cycle. It observes command-bus use, data-bus occupancy, PIM-block
busy masks, BG lifecycle, and rank execution mode. An ineligible BG can be
skipped so another ready BG can proceed.

Legacy rank-wide and CSC BG-targeted execution are mutually exclusive within a
rank. Mode changes pass through a drain state, which prevents legacy and CSC
commands from owning the same physical datapath simultaneously. A targeted
multiply completes no earlier than the cycle after grant.

### 8.5 Masked FP32 SIMD support

CSC columns are processed with the existing eight-lane PIM SIMD datapath. The
extension provides a single `valid_count` contract for masked `MUL` operations:

- Full chunks execute all eight lanes.
- Tail chunks execute only lanes `[0, valid_count)`.
- Inactive lanes do not read operands, perform arithmetic, modify destination
  state, or emit partials.
- The completed physical PIM-block result is the only source used to construct
  BGA input partials.

This preserves the PIM block as the production arithmetic endpoint while
preventing 32-byte alignment padding from becoming sparse work.

### 8.6 Bank Group Accumulator

Each global BG owns a bounded associative BGA containing tagged entries of:

```text
valid, reserved, row_idx, FP32 value, insertion age,
generation, source stream, contribution count
```

For every incoming `{row_idx, value}` partial, the BGA compares the row tag
against its valid entries. A hit performs an ordered FP32 merge. A miss inserts
into a free entry; if the structure is full, a deterministic victim is emitted
before the new row is inserted. After the descriptor producer finishes, final
drain emits all remaining valid entries.

The BGA explicitly models input queues, tag-comparison width and latency, FP32
add latency, accumulator capacity, an output queue, lookup/merge/insert
operations, and downstream stalls. Input batches and outputs carry generation
and sequence identities so retry or backpressure cannot duplicate a
contribution.

The configuration is parameterized through `CSCBGAConfig`. Defaults are 16
input-queue entries, 16 accumulator entries, compare width 16, one-cycle
compare, one-cycle add, and 16 output-queue entries. The observable full-run
test deliberately uses four accumulator entries and a four-entry output queue
as a small bounded configuration; dedicated BGA tests exercise capacity
eviction and pressure. These values are simulator configuration choices, not
fixed silicon dimensions.

### 8.7 Partial-result writeback and host return path

The BGA output destination implements a reserve/accept/commit protocol. If the
destination lacks capacity, the BGA front record remains stable and the stall
propagates upstream. Once accepted, each `{uint32 row_idx, float32 value}` record
enters its BG-local four-record packer.

The modeled result path adds:

- bounded pending write-burst queues per BG;
- configurable write latency and in-flight limits;
- deterministic rank-local write arbitration;
- topology-sized resident burst buffers with reservation at write issue;
- configurable read latency and per-channel in-flight limits;
- deterministic channel-local read arbitration;
- a bounded host return queue with capacity reserved at read issue; and
- a bounded FP32 reduction queue with configurable issue width and latency.

The full-run configuration uses two pending bursts per BG, one in-flight write
per BG, one write issue per rank per cycle, two-cycle write latency, one read
issue and one in-flight read per channel, three-cycle read latency, a two-burst
host return queue, and a four-record reduction queue. The general matrix test
sizes resident capacity from the maximum per-BG NNZ requirement before launch.

### 8.8 Lifecycle, backpressure, and observability

Every BG has independent run, flush, drain, error-drain, error, and reset
states. Accepted work drains normally; it is never force-deleted. Errors are
sticky and prevent completion or result validity. Backpressure can propagate
from host reduction through return queues, resident buffers, write packers,
BGA output, and finally the descriptor engine.

Hardware-model counters expose per-BG ready/executing/wait/grant cycles,
targeted operations, BGA lookup/merge/eviction behavior, write/read traffic,
queue occupancy, stalls, reduction activity, and phase-completion cycles. The
end-to-end completion predicate requires every hardware-model stage and all
record/contribution/byte conservation checks to agree.

### 8.9 Modeling boundary

The extension is a cycle-stepped architectural model, but not every component
is claimed as a fabricated hardware block. In particular:

- BGA, BG-targeted execution, bounded queues, arbitration, latency, ownership,
  and backpressure are explicitly modeled.
- Partial-result writeback models transaction timing and resident ownership; it
  does not persist payloads in a physical DRAM cell array.
- Final indexed reduction is currently a deterministic host-side FP32 model,
  not a logic-die Global Accumulator.
- TSV/internal-bus timing, CPU instruction timing, streaming readback, a
  hardware Global Accumulator, and the complete SparsePIM+ architecture are not
  implemented.

Relevant implementation files are
[`src/csc/CSCDescriptorEngine.h`](src/csc/CSCDescriptorEngine.h),
[`src/BGTargetedOperation.h`](src/BGTargetedOperation.h),
[`src/csc/CSCBankGroupAccumulator.h`](src/csc/CSCBankGroupAccumulator.h), and
[`src/csc/CSCPartialResultPath.h`](src/csc/CSCPartialResultPath.h).

### Contact
* Shin-haeng Kang (s-h.kang@samsung.com)
* Sanghoon Cha (s.h.cha@samsung.com)
* Seungwoo Seo (sgwoo.seo@samsung.com)
* Jin-seong kim (jseong82.kim@samsung.com)

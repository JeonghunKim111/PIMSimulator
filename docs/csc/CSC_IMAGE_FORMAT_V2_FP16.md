# CSC physical image format v2: FP16

## Status and compatibility

Version 2 is a host-exported, host-validated FP16 image contract. It is not yet
accepted by the production descriptor engine. Existing FP32 images remain
version 1; their meaning and loader are unchanged. A v2 parser rejects v1,
unsupported versions, and precision mismatches, while the v1 execution loader
rejects v2.

## Fixed format properties

| Field | Value |
|---|---|
| magic | `SPCSCIMG` |
| format version | 2 |
| endianness | little-endian |
| storage value type | IEEE-754 binary16 |
| value bytes | 2 |
| row-index type/bytes | uint32 / 4 |
| burst/alignment | 32 bytes |
| future SIMD width | 16 |
| values per burst | 16 |
| row indices per burst | 8 |
| descriptor bytes | 32 |
| global BG count | 64 |

All multibyte fields are serialized explicitly in little-endian order. Files
never contain raw `half` or compiler struct object representations.

## Descriptor layout

The existing 32-byte integer descriptor layout is retained:

```text
0x00 uint64 value_offset_bytes
0x08 uint64 row_idx_offset_bytes
0x10 uint32 nnz_count
0x14 uint32 x_slot
0x18 uint32 original_col
0x1c uint32 global_bg_id
```

Offsets are BG-local. A descriptor exists for every nonempty column; empty
columns retain logical dimension and x data but have no descriptor.

## Streams and allocation

For BG numbers 00 through 63:

```text
bg_NN_values_fp16.bin
bg_NN_row_idx_u32.bin
bg_NN_descriptors.bin
bg_NN_x_permutation.bin
```

Every nonempty column begins at a 32-byte boundary in both the value and index
streams. Its allocation is `align32(2*nnz)` and `align32(4*nnz)` respectively.
Unused bytes are zero. The permutation contains one uint32 original-column
number per descriptor, in descriptor/x-slot order.

`x_fp16.bin` contains one FP16 value for every original column followed by zero
padding to 32 bytes. This global stream permits exact reconstruction of x,
including empty columns; per-BG permutation retains execution mapping.

## Canonical conversion

The exporter accepts source values and x as double and performs exactly:

```text
double -> half_float::half -> raw uint16 bits -> little-endian bytes
```

There is no intermediate float conversion. Round-to-nearest-even, preserved
subnormals and signed zero, overflow to signed infinity, and half_float NaN
quieting follow the shared M1 arithmetic contract.

## Manifest and publication

`manifest.json` records format/type widths, matrix dimensions and NNZ,
descriptor/BG counts, column ownership, logical/physical/padding bytes for
values/indices/x, descriptor bytes, conversion and special-value statistics,
per-BG file sizes, and FNV-1a-64 checksums. The global x file has its own size
and checksum.

The exporter refuses an existing output directory. It writes every stream,
reopens it to verify exact size and checksum, and publishes the manifest last.
On failure it removes only the directory it created.

## Loader validation

`loadCSCFp16ImageV2` validates the full contract before reconstruction:

- magic, version, endianness, precision, widths and BG count;
- mandatory files, exact sizes, FNV-1a-64 checksums and alignment;
- descriptor ownership/order, x slots, offsets, ranges and NNZ sum;
- row bounds, original-column uniqueness and BG ownership;
- zero value/index/x padding;
- manifest logical/physical/padding/conversion accounting.

It then reconstructs conventional CSC and x as raw uint16 FP16 bits. Values
are compared bit-exactly, not with decimal tolerance.

## Current limitation

The v2 exporter and loader are an M2 host foundation. They deliberately stop
before the production descriptor engine. Version-2 execution, 16-value and
dual-index scheduling, FP16 PIM MUL, FP16 BGA, and FP16 partial transport are
M3 and later work.

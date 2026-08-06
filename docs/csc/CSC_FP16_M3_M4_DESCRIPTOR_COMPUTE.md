# FP16 M3/M4 descriptor and compute execution

## Scope

M3/M4 connects a fully validated FP16 image v2 to native PIMBlock FP16 MUL and
ends at a bounded compute-to-BGA capture interface:

```text
v2 loader/gate -> one active BG descriptor/chunk
-> X, VALUE, INDEX_LOW, optional INDEX_HIGH
-> 16-entry staging -> masked PIMBlock FP16 MUL
-> indexed FP16 partial capture
```

FP16 BGA, partial serialization/writeback/readback, ordered host FP16
reduction, and final y are explicitly outside this milestone.

## Mode and image gate

`CSCFp16ExecutionImage::load(directory, FP16_IMAGE_V2)` calls the M2 host
loader and only returns after the manifest, every file, checksum, allocation,
descriptor, owner, x slot and padding invariant succeeds. The execution image
has no public constructor from unchecked byte arrays. `CSCFp16DescriptorEngine`
also rejects a non-FP16 PIMBlock.

This is separate from `CSCDescriptorEngine`: FP32 v1 continues to use 8 lanes,
one value and one index burst, FP32 BGA and its existing timing/counters.

## Descriptor and request FSM

```text
IDLE -> FETCH_DESCRIPTOR -> ISSUE_REQUESTS -> WAIT_OPERANDS
     -> SIMD_MUL -> EMIT_PARTIALS -> ADVANCE_CHUNK
     -> ISSUE_REQUESTS (remaining NNZ)
     -> NEXT_DESCRIPTOR -> FETCH_DESCRIPTOR or DONE
```

Any identity, descriptor, or protocol failure enters sticky `ERROR`.

The FP32 issue policy is preserved: one request is attempted per engine tick.
FP16 issue order is X (first chunk only), VALUE, INDEX_LOW, and INDEX_HIGH when
needed. A rejected request remains unchanged and retries before later slots.
Only one chunk is active; there is no prefetch or descriptor interleaving.

Each slot records the full `CSCFp16Request`: request id, kind, global BG,
descriptor id, chunk id, and stream offset. Completion may arrive in any order,
but must match that identity exactly. Duplicate, stale, wrong-descriptor,
wrong-chunk, wrong-kind and wrong-offset completions fail instead of populating
a different operand slot.

## Chunk and independent streams

`valid_count=min(remaining_nnz,16)`. Every nonempty chunk requests one 32-byte
FP16 value burst. Lanes 1-8 request one index burst; lanes 9-16 require a second.

```text
after chunk:
  value_stream_offset += 32
  index_stream_offset += valid_count > 8 ? 64 : 32
```

Descriptor bases remain the M2 column-aligned offsets. For NNZ 17 the relative
requests are values `{0,32}` and indices `{0,32,64}`.

## X addressing and staging

The M2 image stores global x in original-column order. Its address and lane are:

```text
x_burst = original_col / 16
x_lane  = original_col % 16
x_byte_address = 32 * x_burst
```

The descriptor's BG-local `x_slot` is retained and validated by its permutation
stream, but global x addressing uses `original_col`. X is loaded once per
descriptor and reused across its chunks.

Staging contains 16 half values, 16 uint32 rows and one half scalar. VALUE maps
lanes 0-15; INDEX_LOW maps rows 0-7; INDEX_HIGH maps rows 8-15. Upper rows are
not required or read for a short chunk. Correctness never depends on padding.

## Native FP16 multiplication

The engine receives one `PIMBlock(FP16)` and calls its masked
`mul(result, values, scalar, valid_count)` operation. This uses the simulator's
existing half_float arithmetic and 16-lane datapath; there is no replacement
software multiplication loop. Only valid result lanes become events. The
expected test product uses `cscFp16Mul()` and compares raw uint16 bits (NaN by
the numerical contract).

The current single-block policy is unchanged. No second PIM block, different
round robin, wider command issue, or shorter FP16 ALU latency is introduced.

## Partial event and capture handshake

The internal event is:

```text
row_idx, fp16 value_bits,
global_bg_id, descriptor_id, chunk_id, lane_id
```

The identity fields are diagnostic sideband. This is not the future eight-byte
serialized transport record. Events are offered in descriptor, chunk, valid
lane order to `CSCFp16PartialSink`. `CSCFp16BoundedCaptureSink` has finite
capacity plus an enable/ready gate. While not ready, the engine remains on the
same lane and increments sink backpressure cycles; after release it accepts
that event exactly once.

## Counters and invariants

The engine exposes descriptor/NNZ counts, logical x loads and physical x
requests, value, index-low/high/total requests, chunks, active/invalid lanes,
generated/emitted partials, operand waits, request retries and sink stalls.

```text
value requests      = sum ceil(nnz_column/16)
index requests      = sum ceil(nnz_column/8)
index-high requests = sum_chunks [valid_count > 8]
chunks              = sum ceil(nnz_column/16)
active lanes        = generated partials = emitted partials = NNZ
x requests          = nonempty descriptor count
```

The boundary fixture has 169 NNZ, ten descriptors, 15 chunks/value requests,
15 low plus ten high index requests, 71 invalid lanes and 169 emitted events.

## M5 interface requirements and limitations

M5 must implement an FP16 BGA sink with the same ready/accept guarantee and
consume `row_idx + value_bits` without widening to FP32. It must define FP16
ADD/merge, queue entry state, eviction/drain order, batch or per-event ingress
rate and timing. The bounded capture sink is validation infrastructure, not
the final accumulator.

The current engine's memory callback is a typed production seam, while tests
supply validated image bursts and permute completions. M3/M4 does not yet wire
v2 requests into `CSCNativeExecution`/DRAMSim address arbitration; doing so
without adding prefetch or changing issue width remains an integration item
before performance claims are made.

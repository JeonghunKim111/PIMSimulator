# CSC M7 FP16 numerical contract

## Scope and architectural path

The current M7 architecture has no logic-die global accumulator and no dense
PIM-side final-y buffer. The FP16 conversion target is:

```text
FP16 physical image
-> descriptor execution and FP16 MUL
-> BG-local FP16 BGA
-> indexed FP16 partial writeback/readback
-> ordered host FP16 reduction
-> final_y_fp16
```

M1 supplies shared arithmetic and a conventional sequential reference. It does
not connect FP16 to the production descriptor, BGA, or transport path.

## Confirmed semantics

All items below are **[CONFIRMED]** by `CSCFp16Test` and the M0
`FP16SemanticsCharacterizationTest`.

| Property | Contract |
|---|---|
| implementation | `half_float::half`, `lib/half.h` 2.1.0 |
| storage | IEEE-754 binary16, explicit uint16 bits |
| rounding | round-to-nearest-even |
| subnormal | preserved; no flush-to-zero mode |
| underflow | rounds to a subnormal or signed zero |
| overflow | signed infinity |
| infinity | preserved with IEEE operation behavior |
| signed zero | conversion/product sign preserved; existing ADD behavior used |
| NaN | remains NaN and follows half_float quieting semantics |
| finite/Inf/zero validation | raw uint16 bit-exact |
| NaN validation | classification/quieting; payload and sign are not architectural requirements |

The common API is in `src/csc/CSCFp16.h`. Bit conversion uses the simulator's
existing `fp16i` representation. Image bytes are emitted and consumed through
explicit little-endian uint16 helpers; neither `half` objects nor structs are
dumped as compiler object representations.

## Arithmetic boundaries

MUL and ADD are separate binary16 operations. FMA is forbidden for this
contract:

```cpp
const CSCFp16 product = cscFp16Mul(lhs, rhs);
return cscFp16Add(accumulator, product);
```

Thus a multiplication result is materialized as FP16 before a later ADD, and
every accumulator update is rounded back to FP16. A characterization vector
whose separated result differs from `half_float::fma` freezes this boundary.

The same primitives are intended for image conversion, future PIM arithmetic,
BGA arithmetic, ordered host reduction, and references. The M2 canonical
source conversion is direct `double -> half -> uint16 bits`; it never inserts
an intermediate float conversion.

## Reference layers

FP16 addition is order-dependent. References therefore have distinct roles:

1. FP64 SpMV is only an error oracle.
2. `cscFp16SequentialSpMV` consumes exported FP16 value/x bits, traverses
   conventional CSC order, and rounds after every MUL and ADD. It is an
   algorithmic FP16 reference.
3. A future architectural trace-replay reference must follow descriptor, BGA
   eviction/drain, partial delivery, and ordered host-reduction events. It may
   differ bitwise from the sequential reference.

M1 implements only layer 2. It does not invent BGA or transport event order.

## Remaining decisions

- [OPEN] Define the exact architectural event trace exposed by M3-M6 for
  bit-exact replay.
- [OPEN] Decide the supported compiler/release matrix if F16C or fast-math is
  ever enabled. The current build uses neither.

NaN payload preservation, separate MUL/ADD, and the absence of logic-die GA are
not open decisions.

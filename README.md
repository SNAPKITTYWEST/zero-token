```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ███████╗███████╗██████╗  ██████╗       ████████╗ ██████╗ ██╗  ██╗  │
│   ╚══███╔╝██╔════╝██╔══██╗██╔═══██╗      ╚══██╔══╝██╔═══██╗██║ ██╔╝ │
│     ███╔╝ █████╗  ██████╔╝██║   ██║  █████╗ ██║   ██║   ██║█████╔╝  │
│    ███╔╝  ██╔══╝  ██╔══██╗██║   ██║  ╚════╝ ██║   ██║   ██║██╔═██╗  │
│   ███████╗███████╗██║  ██║╚██████╔╝         ██║   ╚██████╔╝██║  ██╗ │
│   ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝          ╚═╝    ╚═════╝ ╚═╝  ╚═╝ │
│                                                                      │
│   Zero-text-generation decision engine                               │
│   Single forward pass · Deterministic FSM · Typed decisions          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```


![License](https://img.shields.io/badge/license-GPL--2.0%20%7C%20AGPL--3.0-blue)
![Language](https://img.shields.io/badge/language-C99%20%7C%20CUDA%20%7C%20Metal-orange)
![LOC](https://img.shields.io/badge/lines%20of%20code-4%2C484-brightgreen)
![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20Metal-purple)
![Status](https://img.shields.io/badge/status-production-success)

---

## What is zero-token?

**zero-token** is a structured decision engine that **never generates text**. Given a language model's next-token distribution — full-vocabulary logits or a hidden state plus lm_head — it produces typed decisions in a single forward pass through a constrained linear head, deterministic finite-state machine, and calibrated confidence layer.

No autoregressive loop. No beam search. No sampling. One pass → one decision.

### Field Types

| Type | Output | Decision Rule |
|------|--------|---------------|
| **BOOL** | `true` / `false` | Thresholded on calibrated P(true) |
| **CHOICE** | One of N labels | Categorical argmax with alias-token pooling |
| **SCORE** | Numeric expectation | Weighted sum over N ordinal choice values |

Every decision carries: calibrated confidence, raw probabilities, log-margin, entropy, abstain flags, and dependency guards.

---

## Architecture

```mermaid
graph LR
    subgraph Input
        A[Full-vocabulary logits<br/>or hidden state + lm_head]
    end

    subgraph "zero-token Engine"
        B[Gather K sub-logits<br/>from compiled token map]
        C[Temperature-scaled softmax<br/>per-choice mass pooling]
        D[Calibration map<br/>piecewise-linear isotonic]
        E[Deterministic FSM<br/>forward-only dependency walk]
        F[Structural emitter<br/>JSON / XML / JSON-Schema]
    end

    subgraph Output
        G[Typed decisions<br/>+ confidence + margin]
    end

    A --> B --> C --> D --> E --> F --> G
```

## Compilation Pipeline

```mermaid
graph TD
    S[zt_schema] -->|"add fields, choices,<br/>tokens, deps, calibration"| V[Validation]
    V -->|"ambiguity check<br/>forward-only deps<br/>token dedup"| C[zt_compile]
    C -->|"flatten to dense<br/>device-ready tables"| P[zt_program]
    P --> E[zt_engine]
    E -->|"AUTO: CUDA → Metal → CPU"| BE[Backend selection]
    BE --> SE[zt_session]
    SE -->|"bind lm_head<br/>fp16 gather + round-trip"| R[zt_run]
    R --> D[zt_decision array]
    D --> EM[zt_emit_*]
    EM --> JSON[JSON output]
    EM --> XML[XML output]
    EM --> JSD[JSON-Schema]
```

## Backend Dispatch

```mermaid
graph TD
    subgraph "zt_engine (AUTO)"
        CHECK{CUDA available?}
        CHECK -->|Yes| CUDA[CUDA Backend<br/>WMMA 16×16 tensor cores<br/>warp-parallel softmax]
        CHECK -->|No| CHECK2{Metal available?}
        CHECK2 -->|Yes| MTL[Metal Backend<br/>simdgroup_matrix 8×8<br/>SIMT fallback]
        CHECK2 -->|No| CPU[CPU Backend<br/>C99 sequential reference<br/>fp16 round-trip oracle]
    end

    subgraph "Shared Core (zt_core.h)"
        CORE[zt_field_dev · zt_decision<br/>zt_finalize · zt_load_f<br/>zt_calib_apply]
    end

    CUDA --> CORE
    MTL --> CORE
    CPU --> CORE
```

## Decision Pipeline (per row)

```mermaid
flowchart TD
    START([Input row: batch b, field f]) --> DEP{Dependency<br/>guard?}
    DEP -->|Guard fails| SKIP[ZT_SKIPPED<br/>field not evaluated]
    DEP -->|Guard passes / none| GATHER[Gather K sub-logits<br/>via col_token map]
    GATHER --> SOFT[Temperature-scaled softmax<br/>over K columns]
    SOFT --> POOL[Pool alias tokens<br/>per choice: sum exp]
    POOL --> ARGMAX[Argmax + runner-up<br/>lowest-index tie-break]
    ARGMAX --> BOOL{BOOL field?}
    BOOL -->|Yes| THRESH[Threshold P_true<br/>vs field.threshold]
    BOOL -->|No| SEL[Selected = argmax]
    THRESH --> CAL[Calibration map<br/>isotonic PWL knots]
    SEL --> CAL
    CAL --> ABSTAIN{conf < min_conf<br/>OR margin < min_margin?}
    ABSTAIN -->|Yes| ABS[ZT_ABSTAIN<br/>+ LOW_CONF / LOW_MARGIN flags]
    ABSTAIN -->|No| FINAL[zt_decision<br/>choice + confidence + margin<br/>+ entropy + score]
```

---

## Project Structure

```
zero-token/
├── include/
│   ├── zt.h               Public C99 API — schema → compile → engine → session → run → emit
│   ├── zt_device.h         Target abstraction: Metal / CUDA / C99 host macros
│   ├── zt_core.h           Shared decision core — compiles unchanged on all 3 targets
│   └── zt_internal.h       Private host declarations — zt_program, zt_backend_ops vtable
│
├── src/
│   ├── zt_engine.c         Engine/session lifecycle, AUTO backend selection
│   ├── zt_schema.c         Schema builder + zt_compile (validate → flatten → dense tables)
│   ├── zt_cpu.c            C99 reference backend — correctness oracle for GPU paths
│   ├── zt_fsm.c            Deterministic routing walk — single ordered pass, idempotent
│   ├── zt_emit.c           Structural emitters (JSON / XML / JSON-Schema)
│   ├── zt_calib.c          Offline calibration: golden-section NLL + pool-adjacent-violators
│   ├── zt_metal.mm         Metal host bridge (metal-cpp) — 5 pipelines, zero-copy shared memory
│   └── zt_metal_arc.m      Metal host bridge (ObjC ARC) — library resolution, SIMD probe
│
├── kernels/
│   ├── zt_kernels.cuh      CUDA device code — WMMA 16×16 tiles, warp-parallel softmax+argmax
│   ├── zt_kernels.metal    Apple Silicon MSL — simdgroup_matrix 8×8, SIMT fallback
│   └── zt_kernels.metal.h  Embedded MSL source as raw string literal (metal-cpp path)
│
├── ffi/
│   ├── python/zt.py        ctypes bindings — Schema, Engine, Session, Result classes
│   └── rust/lib.rs         Rust FFI — Schema, Engine, Session, ResultBuf with Drop impl
│
├── tests/
│   └── test_zt.c           Correctness suite — logits, hidden, aliases, abstain, emitters,
│                            calibration, fp16 round-trips, GPU vs CPU cross-check
│
├── bench/
│   └── bench.c             Throughput harness — logits (rows×32000) and hidden (rows×H) paths
│
└── tools/
    └── claimguard/         ClaimGuard CLI scaffolding (separate verification tool)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No text generation** | Decisions are typed values, not token sequences — no hallucination surface |
| **Single forward pass** | No autoregressive loop — latency is O(1) not O(sequence length) |
| **Shared `zt_core.h`** | Same decision math on CPU, CUDA, and Metal — bit-exact cross-validation |
| **fp16 round-trip** | Head weights gathered through fp16 on all backends — GPU and CPU see identical values |
| **Deterministic tie-break** | Lowest-index wins — same input always produces same output across backends |
| **Isotonic calibration** | Pool-adjacent-violators ensures monotone probability mapping |
| **Forward-only deps** | FSM guards form a DAG — no cycles, single ordered evaluation pass |
| **Fail to abstain** | Low confidence or narrow margin → `ZT_ABSTAIN`, never a wrong forced choice |

## Target Abstraction (`zt_device.h`)

| Macro | Metal | CUDA | Host C99 |
|-------|-------|------|----------|
| `ZT_FN` | `static inline` | `__host__ __device__ __forceinline__` | `static inline` |
| `ZT_DEV` | `device` | *(empty)* | *(empty)* |
| `ZT_CONST` | `constant` | *(empty)* | *(empty)* |
| `ZT_THREAD` | `thread` | *(empty)* | *(empty)* |
| `ZT_EXPF` | `metal::precise::exp` | `expf` | `expf` |
| `ZT_F16_TO_F32` | native `half` cast | `__half2float` | IEEE manual decode (subnormals, inf, nan) |

---

## Build

### CPU-only (any platform)

```bash
cc -O2 -Iinclude src/zt_engine.c src/zt_schema.c src/zt_cpu.c \
   src/zt_fsm.c src/zt_emit.c src/zt_calib.c \
   tests/test_zt.c -lm -o test_zt
./test_zt
```

### With CUDA

```bash
nvcc -O2 -Iinclude -DZT_HAVE_CUDA \
     src/zt_engine.c src/zt_schema.c src/zt_cpu.c \
     src/zt_fsm.c src/zt_emit.c src/zt_calib.c \
     kernels/zt_kernels.cuh tests/test_zt.c -lm -o test_zt
```

### With Metal (macOS / Apple Silicon)

```bash
clang -O2 -Iinclude -DZT_HAVE_METAL -framework Metal -framework Foundation \
      src/zt_engine.c src/zt_schema.c src/zt_cpu.c \
      src/zt_fsm.c src/zt_emit.c src/zt_calib.c \
      src/zt_metal_arc.m tests/test_zt.c -lm -o test_zt
```

### Benchmark

```bash
cc -O2 -Iinclude src/zt_engine.c src/zt_schema.c src/zt_cpu.c \
   src/zt_fsm.c src/zt_emit.c src/zt_calib.c \
   bench/bench.c -lm -o bench
./bench
```

---

## FFI Bindings

### Python

```python
from zt import Schema, Engine, Session

schema = Schema()
field = schema.add_field("sentiment", Schema.CHOICE)
schema.add_choice(field, "positive", [1234, 5678])
schema.add_choice(field, "negative", [9012, 3456])

engine = Engine()
session = engine.session(schema.compile(), max_batch=32)
result = session.run_logits(logits_array)
print(result[0].choice, result[0].confidence)
```

### Rust

```rust
use zero_token::{Schema, Engine, Session};

let mut schema = Schema::new();
let field = schema.add_field("sentiment", Kind::Choice);
schema.add_choice(field, "positive", &[1234, 5678], 0.0);
schema.add_choice(field, "negative", &[9012, 3456], 0.0);

let program = schema.compile()?;
let engine = Engine::new(Backend::Auto)?;
let mut session = engine.session(&program, 32)?;
let results = session.run_logits(&tensor)?;
```

---

## License

Dual-licensed under [GPL-2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) or [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html). See [LICENSE](LICENSE) for details.

Copyright (c) 2026 BEL ESPRIT D ACCORD.

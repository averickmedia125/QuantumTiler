# Quantum Barrier Mathematics

This document explains the mathematical foundations of QuantumTiler's adaptive tiling algorithm, which draws inspiration from WKB-style quantum tunneling through a golden-ratio-based barrier potential.

## Core Concept

Traditional matrix multiplication tiling uses fixed tile sizes, ignoring runtime system state. QuantumTiler treats the CPU's thermal/power/latency state as an **energy level** and uses quantum tunneling mathematics to derive optimal tile sizes that "tunnel through" performance barriers.

## The Barrier Potential

We define a barrier potential function:

```
V(x) = φ^(-|2x|/δ) - 1
```

Where:
- `φ = (1 + √5) / 2 ≈ 1.618` — the golden ratio
- `δ = ln(n)` — scaling factor based on matrix size n
- `x` — position coordinate

This potential has:
- **Maximum at x=0**: V(0) = 0
- **Asymptotic decay**: V(±∞) → -1
- **Golden ratio scaling**: The ratio determines barrier "thickness"

## Energy State Mapping

System state is mapped to a negative energy E ∈ (-1, 0):

```
E_total = E_latency + E_temperature + E_power
```

### Latency Component
```
E_latency = -latency_cycles / max_latency
         = -cycles / 200
```
Higher memory latency → more negative energy → smaller tiles.

### Temperature Component
```
E_temperature = -0.3 × (T - 40°C) / (100°C - 40°C)
```
Hotter CPU → more negative energy → smaller tiles (reduced thermal pressure).

### Power Component
```
E_power = -0.2 × (W - TDP/2) / (TDP/2)
```
Higher power draw → more negative energy → smaller tiles (reduced power density).

## The Barrier Exponent

The WKB-style barrier exponent quantifies "difficulty" of tunneling:

```
B(E) = (2√2 / 3) × δ × |E|^1.5 / ln(φ)
     ≈ 1.96 × δ × |E|^1.5
```

Where:
- `δ = ln(n)` — increases with matrix size
- `|E|^1.5` — steeper penalty for stressed systems

### Example Calculation (n=2048)
```
E_latency     = -20/200                      = -0.100
E_temperature = -0.3×(50-40)/(100-40)        = -0.050
E_power       = -0.2×(40-32.5)/(32.5)        = -0.046
─────────────────────────────────────────────────────────
E_total       = -0.196

δ = ln(2048) = 7.625
B = 1.96 × 7.625 × |−0.196|^1.5 = 1.30
```

## Tunneling Probability

The probability of "tunneling through" the barrier:

```
T(E) = exp(-2B)
```

For E = -0.196:
```
T = exp(-2 × 1.30) = exp(-2.6) ≈ 0.074
```

Low T indicates difficulty, triggering task splitting when T < 0.3 (threshold).

## Tile Size Derivation

The optimal tile size emerges from the barrier exponent:

```
tile = scale × exp(-B) × √(cache_size)
```

Where:
- `scale = 32` — base scaling constant
- `cache_size = 256 KB` — L2 cache (target level)
- Result clamped to [min_tile, max_tile] = [32, 128]

### Example
```
tile = 32 × exp(-1.30) × √256
     = 32 × 0.27 × 16
     = 140 → clamped to 128
```

## Adaptive Splitting Logic

When tunneling is difficult (T < split_threshold), tasks split:

```cpp
if (T < 0.3 && depth < max_depth && tile > min_tile) {
    // Split task into 4 sub-tasks (i,j only, not k)
    // Each sub-task has half the tile size
}
```

This allows fine-grained adaptation under stress without race conditions.

## Why Golden Ratio?

The golden ratio φ provides:

1. **Optimal scaling behavior**: φ^n grows smoothly, avoiding sharp transitions
2. **Self-similarity**: Splitting by φ maintains proportional structure
3. **Historical significance**: Appears in natural optimization (phyllotaxis, etc.)
4. **Numerical stability**: ln(φ) ≈ 0.481 provides balanced barrier widths

## Physical Intuition

| System State | Energy E | Barrier B | Tile Size | Behavior |
|--------------|----------|-----------|-----------|----------|
| Cool, idle | -0.10 | 0.5 | 128 | Large tiles, max throughput |
| Warm, active | -0.30 | 2.0 | 64 | Medium tiles, balanced |
| Hot, stressed | -0.60 | 4.5 | 32 | Small tiles, thermal relief |

## Comparison to Classical Approaches

| Approach | Tile Selection | Adaptation |
|----------|----------------|------------|
| ATLAS | Empirical search at compile time | None |
| OpenBLAS | Fixed per architecture | None |
| Intel MKL | Runtime heuristics | Binary (small/large) |
| **QuantumTiler** | Continuous barrier-derived | Real-time, physics-based |

## References

1. Griffiths, *Introduction to Quantum Mechanics* — WKB approximation
2. Goldberg, *The Golden Ratio* — Mathematical properties of φ
3. Goto & Van De Geijn, *High-Performance GEMM* — Tiling fundamentals
4. Intel i7-7700 Datasheet — Cache hierarchy and specifications

---

*Author: Timothy McGirl (Pedesis TM)*  
*Contact: tim@leuklogic.com*

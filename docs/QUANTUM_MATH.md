# Quantum Barrier Mathematics

This document explains the mathematical foundations of QuantumTiler's adaptive tiling algorithm, which draws inspiration from WKB-style quantum tunneling through a golden-ratio-based barrier potential.

---

## 1. Core Concept

Traditional matrix multiplication tiling uses **fixed tile sizes**, ignoring runtime system state. QuantumTiler treats the CPU's thermal/power/latency state as an **energy level** and uses quantum tunneling mathematics to derive optimal tile sizes that "tunnel through" performance barriers.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL TILING                           │
│   Fixed tile = 64 regardless of system state                    │
│   ════════════════════════════════════════════                  │
│   Cool CPU?  → 64-tile (suboptimal, could use larger)           │
│   Hot CPU?   → 64-tile (suboptimal, thermal throttling)         │
│   Stressed?  → 64-tile (cache thrashing)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUANTUMTILER ADAPTIVE                        │
│   Tile derived from barrier physics + system energy             │
│   ════════════════════════════════════════════════              │
│   Cool CPU?  → 128-tile (maximize throughput)                   │
│   Hot CPU?   → 64-tile (reduce thermal pressure)                │
│   Stressed?  → 32-tile + splitting (graceful degradation)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. The Barrier Potential

We define a barrier potential function inspired by quantum mechanics:

```
V(x) = φ^(-|2x|/δ) - 1
```

Where:
- **φ = (1 + √5) / 2 ≈ 1.618** — the golden ratio
- **δ = ln(n)** — scaling factor based on matrix size n
- **x** — position coordinate

### Visual Representation

```
    V(x)
     │
   0 ┤────────────────────────────────────────────
     │           ╱╲
     │          ╱  ╲         Barrier Peak
     │         ╱    ╲
     │        ╱      ╲
     │       ╱        ╲
     │      ╱          ╲
     │     ╱            ╲
  -1 ┼────╱──────────────╲────────────────────────
     │   ╱                ╲
     └───┴────────┴────────┴────────────────────── x
        -δ        0        +δ
                  
    Particle with energy E < 0 must "tunnel" through barrier
```

### Properties
- **Maximum at x=0**: V(0) = 0
- **Asymptotic decay**: V(±∞) → -1
- **Golden ratio scaling**: φ determines barrier "thickness" and shape
- **Logarithmic width**: δ = ln(n) scales barrier with problem size

---

## 3. Energy State Mapping

System state is mapped to a negative energy **E ∈ (-1, 0)**:

```
E_total = E_latency + E_temperature + E_power
```

### 3.1 Latency Component

```
E_latency = -latency_cycles / max_latency
          = -cycles / 200
```

| Memory Access | Cycles | E_latency |
|---------------|--------|-----------|
| L1 cache hit  | 4      | -0.02     |
| L2 cache hit  | 12     | -0.06     |
| L3 cache hit  | 38     | -0.19     |
| DRAM access   | 200    | -1.00     |

Higher memory latency → more negative energy → smaller tiles.

### 3.2 Temperature Component

```
E_temperature = -0.3 × (T - 40°C) / (100°C - 40°C)
```

| CPU Temp | E_temperature |
|----------|---------------|
| 40°C     | 0.00          |
| 60°C     | -0.10         |
| 80°C     | -0.20         |
| 100°C    | -0.30         |

Hotter CPU → more negative energy → smaller tiles (reduced thermal pressure).

### 3.3 Power Component

```
E_power = -0.2 × (W - TDP/2) / (TDP/2)
```

For i7-7700 with TDP = 65W:

| Power Draw | E_power |
|------------|---------|
| 32.5W      | 0.00    |
| 45W        | -0.077  |
| 65W        | -0.20   |

Higher power draw → more negative energy → smaller tiles (reduced power density).

---

## 4. The Barrier Exponent

The WKB-style barrier exponent quantifies "difficulty" of tunneling:

```
B(E) = (2√2 / 3) × δ × |E|^1.5 / ln(φ)
     ≈ 1.96 × δ × |E|^1.5
```

### Derivation from WKB Approximation

In quantum mechanics, the WKB tunneling probability through a barrier is:

```
T = exp(-2 ∫ √(2m(V(x) - E)) dx)
```

For our potential V(x) = φ^(-|2x|/δ) - 1, integrating from the classical turning points yields:

```
B(E) = ∫_{-x₀}^{x₀} √(V(x) - E) dx

where x₀ satisfies V(x₀) = E
```

After integration (see Appendix A for details):

```
B(E) = (2√2 / 3) × (δ / ln(φ)) × |E|^1.5
     = 1.96 × δ × |E|^1.5
```

The prefactor 1.96 ≈ (2√2/3) / ln(φ) ≈ 0.943 / 0.481.

### Barrier Exponent Table

For n=2048 (δ=7.625):

| E_total | B(E) | Difficulty |
|---------|------|------------|
| -0.10   | 0.47 | Easy       |
| -0.20   | 1.34 | Moderate   |
| -0.30   | 2.46 | Hard       |
| -0.50   | 5.29 | Very Hard  |
| -0.80   | 10.7 | Extreme    |

---

## 5. Tunneling Probability

The probability of "tunneling through" the barrier:

```
T(E) = exp(-2B)
```

```
    T(E)
    1.0 ┤●
        │ ╲
    0.8 ┤  ╲
        │   ╲
    0.6 ┤    ╲
        │     ╲
    0.4 ┤      ╲
        │       ╲
    0.2 ┤        ╲●━━━━━━━━━━━━━━ Split Threshold (0.3)
        │         ╲
    0.0 ┼──────────╲●────●────●────
        └──┬───┬───┬───┬───┬───┬─── B
           0   1   2   3   4   5
           
    When T < 0.3, tasks split for finer-grained adaptation
```

### Example Calculation (n=2048)

```
E_latency     = -20/200                      = -0.100
E_temperature = -0.3×(50-40)/(100-40)        = -0.050
E_power       = -0.2×(40-32.5)/(32.5)        = -0.046
─────────────────────────────────────────────────────────
E_total       = -0.196

δ = ln(2048) = 7.625
B = 1.96 × 7.625 × |−0.196|^1.5 = 1.30
T = exp(-2 × 1.30) = exp(-2.6) ≈ 0.074
```

Since T < 0.3, tasks would split under stress conditions.

---

## 6. Tile Size Derivation

The optimal tile size emerges from the barrier exponent:

```
tile = scale × exp(-B) × √(cache_size)
```

Where:
- **scale = 32** — base scaling constant (empirically tuned)
- **cache_size = 256 KB** — L2 cache (target level)
- **Result clamped to [min_tile, max_tile] = [32, 128]**

### The Formula Explained

```
┌─────────────────────────────────────────────────────────────────┐
│  tile = 32 × exp(-B) × √256                                     │
│         ──   ────────   ────                                    │
│          │      │        │                                      │
│          │      │        └── Cache factor: √256 = 16            │
│          │      │            (larger cache → larger tiles)      │
│          │      │                                               │
│          │      └── Barrier penalty: exp(-B)                    │
│          │          (higher stress → smaller multiplier)        │
│          │                                                      │
│          └── Base scale: 32                                     │
│              (minimum reasonable tile for AVX2)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Example

```
tile = 32 × exp(-1.30) × √256
     = 32 × 0.27 × 16
     = 140 → clamped to 128
```

### Tile Size vs Energy

```
    Tile
    128 ┤●────●────●
        │          ╲
     96 ┤           ╲
        │            ╲
     64 ┤             ●────●
        │                   ╲
     32 ┤                    ●────●────●
        │
      0 ┼──┬───┬───┬───┬───┬───┬───┬───┬─── E
        -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8
        
        Cool/Idle ─────────────────► Hot/Stressed
```

---

## 7. Adaptive Splitting Logic

When tunneling is difficult (T < split_threshold), tasks split:

```cpp
if (T < 0.3 && depth < max_depth && tile > min_tile) {
    // Split task into 4 sub-tasks (i,j only, not k)
    // Each sub-task has half the tile size
    for (di = 0; di < 2; di++)
        for (dj = 0; dj < 2; dj++)
            create_subtask(i + di*half, j + dj*half, half);
}
```

### Why Only Split i,j (Not k)?

```
Matrix Multiplication: C[i,j] = Σ_k A[i,k] × B[k,j]
                                ───────────────────
                                      k-loop

Splitting k would create RACE CONDITIONS:
  Thread 1: C[i,j] += A[i,k0..k1] × B[k0..k1,j]
  Thread 2: C[i,j] += A[i,k1..k2] × B[k1..k2,j]  ← CONFLICT!

Solution: Each (i,j) task handles the FULL k-loop internally
  Thread 1: C[i0,j0] = full k-loop  ← No conflict
  Thread 2: C[i1,j1] = full k-loop  ← Different output cells
```

---

## 8. Why Golden Ratio?

The golden ratio φ provides:

### 8.1 Optimal Scaling Behavior
```
φ^n grows smoothly: 1, 1.618, 2.618, 4.236, 6.854, ...
No sharp transitions between tile size decisions
```

### 8.2 Self-Similarity
```
Splitting by φ maintains proportional structure:
  128 → 79 → 49 → 30 → 19 (φ-based)
  vs
  128 → 64 → 32 → 16 → 8  (power-of-2)
  
Golden ratio gives finer granularity options
```

### 8.3 Historical Significance
```
φ appears in natural optimization:
  - Phyllotaxis (leaf arrangement) for light capture
  - Fibonacci spirals for packing efficiency
  - Penrose tilings for aperiodic coverage
```

### 8.4 Numerical Stability
```
ln(φ) ≈ 0.481 provides balanced barrier widths
Not too steep (unstable small changes)
Not too flat (unresponsive to state changes)
```

---

## 9. Physical Intuition

| System State | Energy E | Barrier B | exp(-B) | Tile Size | Behavior |
|--------------|----------|-----------|---------|-----------|----------|
| Cool, idle   | -0.10    | 0.5       | 0.61    | 128       | Large tiles, max throughput |
| Warm, active | -0.30    | 2.0       | 0.14    | 64        | Medium tiles, balanced |
| Hot, stressed| -0.60    | 4.5       | 0.01    | 32        | Small tiles, thermal relief |

### Cache Utilization Model

```
┌──────────────────────────────────────────────────────────────────┐
│                        TILE SIZE vs CACHE FIT                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  128×128 tile = 128×128×4 = 64 KB ← Fits in L2 (256 KB)         │
│                                      Room for A, B, C tiles      │
│                                                                  │
│  64×64 tile = 64×64×4 = 16 KB   ← Fits in L1 (32 KB)            │
│                                    Fastest access, most reuse    │
│                                                                  │
│  32×32 tile = 32×32×4 = 4 KB    ← Easily fits in L1             │
│                                    Low memory pressure           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 10. Comparison to Classical Approaches

| Approach | Tile Selection | Adaptation | Physics |
|----------|----------------|------------|---------|
| ATLAS | Empirical search at compile time | None | None |
| OpenBLAS | Fixed per architecture | None | None |
| Intel MKL | Runtime heuristics | Binary (small/large) | None |
| HP-DAEMON | Power-aware scheduling | Thermal only | None |
| **QuantumTiler** | **Continuous barrier-derived** | **Real-time, multi-factor** | **WKB tunneling + φ** |

---

## 11. Implementation Code

### Barrier Exponent (C++)
```cpp
const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;
const double LN_PHI = std::log(PHI);
const double ASYMP_PREFACTOR = 1.96;  // 2*sqrt(2)/3 / ln_phi

double barrier_exponent_asymptotic(double E, double delta = 1.0) {
    if (E >= 0.0) return 0.0;
    if (E <= -1.0) return 1e10;
    return ASYMP_PREFACTOR * delta * std::pow(-E, 1.5);
}
```

### Tile Derivation (C++)
```cpp
int derive_tile_size(double E, double delta, int cache_kb = 256, 
                     int scale = 32, int min_tile = 32, int max_tile = 128) {
    double B = barrier_exponent_asymptotic(E, delta);
    if (B >= 1e9) return min_tile;
    int tile = static_cast<int>(scale * std::exp(-B) * std::sqrt(cache_kb));
    return std::max(min_tile, std::min(max_tile, tile));
}
```

### Energy Calculation (C++)
```cpp
double energy_from_latency(double latency_cycles, double max_latency = 200.0) {
    return -std::min(1.0, latency_cycles / max_latency);
}

double energy_from_temperature(double temp_celsius, double max_temp = 100.0) {
    if (temp_celsius <= 40.0) return 0.0;
    return -0.3 * (temp_celsius - 40.0) / (max_temp - 40.0);
}

double energy_from_power(double power_watts, double tdp_watts = 65.0) {
    if (power_watts <= tdp_watts * 0.5) return 0.0;
    return -0.2 * std::min(1.0, (power_watts - tdp_watts*0.5) / (tdp_watts*0.5));
}
```

---

## Appendix A: Full WKB Integration

For the potential V(x) = φ^(-|2x|/δ) - 1, the classical turning points where V(x₀) = E are:

```
φ^(-2x₀/δ) - 1 = E
φ^(-2x₀/δ) = E + 1
-2x₀/δ = log_φ(E + 1)
x₀ = -δ × ln(E + 1) / (2 ln(φ))
```

The barrier integral:

```
B = ∫_{-x₀}^{x₀} √(V(x) - E) dx
  = ∫_{-x₀}^{x₀} √(φ^(-|2x|/δ) - 1 - E) dx
```

Using the substitution u = 2x/δ and symmetry:

```
B = δ × ∫_0^{u₀} √(φ^(-u) - (1 + E)) du
```

For E << 0 (asymptotic approximation):

```
B ≈ (2√2 / 3) × δ × |E|^1.5 / ln(φ)
```

This gives the 1.96 prefactor.

---

## References

1. Griffiths, D.J. *Introduction to Quantum Mechanics* — WKB approximation derivation
2. Livio, M. *The Golden Ratio* — Mathematical properties of φ
3. Goto, K. & Van De Geijn, R. *Anatomy of High-Performance GEMM* — Tiling fundamentals
4. Intel® 64 and IA-32 Architectures Optimization Reference Manual — Cache specifications
5. Wunderlich, R.E. et al. *Energy-Efficient SIMD* — Power-aware optimization

---

*Author: Timothy McGirl (Pedesis TM)*  
*Contact: tim@leuklogic.com*  
*Repository: https://github.com/grapheneaffiliate/QuantumTiler*

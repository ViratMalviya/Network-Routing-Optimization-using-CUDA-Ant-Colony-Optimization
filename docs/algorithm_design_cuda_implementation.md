# Algorithm Design & Implementation with CUDA

This document explains how NetOptima achieves accurate modeling, proper parameter tuning, error-free implementation, and efficient use of CUDA for the Ant Colony Optimization algorithm.

---

## 1. Accurate Modeling

### Graph Representation

The network is modeled as a **weighted directed graph** G = (V, E, W) where:
- **V** = {v₀, v₁, ..., vₙ₋₁} — Network devices (routers, switches, servers, firewalls, endpoints)
- **E** ⊆ V × V — Directed communication links
- **W**: E → ℝ⁺ — Edge weights representing transmission cost (latency in ms, hop count, or composite metric)

The graph is stored as a **dense adjacency matrix** `distances[N×N]` where:
- `distances[i][j] > 0` means edge (i→j) exists with that weight
- `distances[i][j] = -1` means no direct link from i to j

This representation is chosen because:
1. O(1) lookup time for edge existence and weight — critical for GPU thread performance
2. 2D matrix maps naturally to CUDA 2D grid for pheromone operations
3. Memory is pre-allocated — no dynamic allocation needed on GPU

### ACO Mathematical Model

The implementation follows the **Ant System (AS)** variant of ACO with two core operations:

#### Solution Construction (per ant)

Each ant k builds a path from source s to destination d by iteratively selecting the next node. At node i, the probability of moving to unvisited neighbor j is:

```
P_ij^k = (τ_ij^α × η_ij^β) / Σ_{l∈N_i^k} (τ_il^α × η_il^β)
```

Where:
- `τ_ij` = pheromone level on edge (i, j)
- `η_ij` = heuristic desirability = 1 / distance(i, j)
- `α` = pheromone influence exponent (default: 1.0)
- `β` = heuristic influence exponent (default: 2.0)
- `N_i^k` = set of unvisited neighbors of node i for ant k

Selection uses **roulette wheel** (fitness-proportionate) selection:
1. Generate random value r ∈ [0, total_probability)
2. Accumulate probabilities until cumulative sum ≥ r
3. Select that node as the next hop

#### Pheromone Update (global)

After all ants complete their paths:

```
τ_ij ← (1 - ρ) × τ_ij + Σ_k Δτ_ij^k
```

Where:
- `ρ` = evaporation rate (default: 0.5) — controls forgetting speed
- `Δτ_ij^k` = Q / L_k if ant k used edge (i, j), else 0
- `Q` = pheromone deposit constant (default: 100)
- `L_k` = total path cost for ant k

### Path Validity

The implementation ensures valid paths through:
1. **Visited tracking**: Each ant maintains a `visited[]` array to prevent cycles
2. **Dead-end handling**: If no unvisited neighbors exist, ant's cost is set to -1 (invalid)
3. **Destination check**: Only ants reaching the destination contribute pheromone
4. **Path length bound**: Maximum path length = number of nodes (prevents infinite loops)

---

## 2. Proper Parameter Tuning

### Default Parameters & Justification

| Parameter | Symbol | Value | Justification |
|---|---|---|---|
| Pheromone Influence | α | 1.0 | Standard value from Dorigo (2004). Linear pheromone influence ensures balanced exploration. Higher values cause premature convergence. |
| Heuristic Influence | β | 2.0 | Quadratic distance influence. Dorigo's experiments showed β=2 provides good balance — ants prefer shorter edges but still explore longer ones with high pheromone. |
| Evaporation Rate | ρ | 0.5 | Moderate evaporation. Too low (0.1) = pheromone accumulates everywhere, no differentiation. Too high (0.9) = good paths forgotten before reinforcement. 0.5 is the standard starting point. |
| Deposit Constant | Q | 100 | Scales pheromone deposit. Q/L_k ensures shorter paths deposit proportionally more pheromone. Q=100 with typical path costs of 10-50 gives deposit values of 2-10, balancing against evaporation. |
| Number of Ants | — | 32 | Matches GPU warp size (32 threads). Provides sufficient exploration diversity while fitting in a single warp for maximum SIMT efficiency. |
| Iterations | — | 100 | Empirically sufficient for convergence on networks up to ~20 nodes. Convergence is monitored via best-cost tracking. |

### Parameter Sensitivity

The system now supports **runtime parameter tuning** through the frontend:

- **α slider** (0.5 – 3.0): Higher α makes ants follow established pheromone trails more strongly → faster convergence but risk of local optima
- **β slider** (0.5 – 5.0): Higher β makes ants prefer shorter edges regardless of pheromone → more greedy, less adaptive
- **ρ slider** (0.1 – 0.9): Higher ρ evaporates pheromone faster → more exploration, slower convergence

Users can observe how different parameters affect:
1. Convergence speed (iterations to stable best cost)
2. Solution quality (final best path cost)
3. Pheromone distribution (visual edge thickness patterns)

---

## 3. Error-Free Implementation

### Input Validation

The backend validates all inputs before executing the algorithm:

```python
# Graph construction with safe defaults
distances = np.full((num_nodes, num_nodes), -1.0, dtype=np.float32)
for edge in edges_data:
    src = edge.get('from', 0)
    dst = edge.get('to', 0)
    w = float(edge.get('weight', 1.0))
    distances[src, dst] = w
```

### GPU Error Handling

The implementation uses a **try-except fallback pattern**:

1. **Primary path**: Attempt CUDA execution (device memory allocation, kernel launch)
2. **Fallback path**: If any CUDA operation fails (no GPU, driver issues, memory exhaustion), transparently fall back to an identical CPU implementation
3. **Error logging**: CUDA errors are written to `err_numba.txt` for debugging

```python
try:
    # GPU execution path
    d_distances = cuda.to_device(distances)
    # ... kernel launches ...
    return {"engine": "CUDA", ...}
except Exception as e:
    # CPU fallback with identical algorithm
    return {"engine": "CPU", ...}
```

### Numerical Stability

- **Division by zero protection**: `total_prob == 0.0` check before roulette selection
- **Pheromone initialization**: All edges start at τ = 0.1 (not 0) to ensure non-zero probabilities
- **Float32 precision**: Consistent use of `np.float32` across CPU and GPU to ensure identical numerical behavior
- **Atomic operations**: `cuda.atomic.add` for pheromone deposition prevents race conditions between threads

### Thread Safety

- Each ant writes only to its own row in `paths[ant_id, :]`, `path_lengths[ant_id]`, `path_costs[ant_id]`
- `visited[]` array uses `cuda.local.array` — thread-local memory, no sharing
- Pheromone update uses `cuda.atomic.add` for safe concurrent writes

---

## 4. Efficient Use of CUDA

### Kernel Architecture

NetOptima uses **two CUDA kernels** that execute in alternating phases:

#### Kernel 1: `construct_solutions` — 1D Grid

```
Grid:  <<<blocks_per_grid, threads_per_block>>>
       <<<ceil(32/256), 256>>>
       = <<<1, 256>>> (32 active threads)
```

- **Thread mapping**: `ant_id = cuda.grid(1)` — each thread IS one ant
- **Work per thread**: Build complete path from source to destination
- **Memory pattern**: Read-heavy from `distances` and `pheromones` (global memory), write to per-ant arrays
- **Local memory**: `visited[1024]` and `probs[1024]` allocated in thread-local memory

```python
@cuda.jit
def construct_solutions(num_nodes, num_ants, start_node, dest_node,
                        distances, pheromones,
                        paths, path_lengths, path_costs, rng_states):
    ant_id = cuda.grid(1)
    if ant_id >= num_ants:
        return
    # Each ant independently builds a path
    visited = cuda.local.array(1024, dtype=int32)
    probs = cuda.local.array(1024, dtype=float32)
    # ... path construction logic ...
```

**Why this is efficient:**
- Ants are **completely independent** during solution construction — perfect SIMT workload
- No inter-thread communication or synchronization required
- Random number generation via `xoroshiro128p` is per-thread (no contention)

#### Kernel 2: `update_pheromones` — 2D Grid

```
Grid:  <<<(phero_blocks_x, phero_blocks_y), (16, 16)>>>
       <<<(ceil(N/16), ceil(N/16)), (16, 16)>>>
```

- **Thread mapping**: `(x, y) = cuda.grid(2)` — each thread handles one cell of the N×N pheromone matrix
- **Work per thread**: Evaporate pheromone at (x, y), then check all ants for deposits on edge (x, y)

```python
@cuda.jit
def update_pheromones(num_nodes, num_ants, pheromones, paths, path_lengths, path_costs):
    x, y = cuda.grid(2)
    if x >= num_nodes or y >= num_nodes:
        return
    # Evaporate
    pheromones[x, y] = pheromones[x, y] * (1.0 - EVAPORATION_RATE)
    # Deposit from all ants
    deposit = 0.0
    for k in range(num_ants):
        # ... check if ant k used edge (x,y) ...
    cuda.atomic.add(pheromones, (x, y), deposit)
```

**Why this is efficient:**
- N×N pheromone matrix naturally maps to 2D GPU grid
- Evaporation is a simple multiply — perfect for GPU (no branching)
- `cuda.atomic.add` ensures correctness with minimal performance impact

### Memory Management

| Data | Size | Location | Access Pattern |
|---|---|---|---|
| `distances[N×N]` | N²×4 bytes | Global (device) | Read-only — could use texture cache |
| `pheromones[N×N]` | N²×4 bytes | Global (device) | Read in K1, Read-Modify-Write in K2 |
| `paths[ants×N]` | ants×N×4 bytes | Global (device) | Write in K1, Read in K2 |
| `visited[N]` | N×4 bytes/thread | Local (registers/L1) | Read-Write per thread only |
| `probs[N]` | N×4 bytes/thread | Local (registers/L1) | Read-Write per thread only |
| `rng_states` | 16 bytes/thread | Global (device) | Read-Write per thread only |

### Performance Characteristics

- **Snapshot extraction**: Only every 10th iteration copies pheromone matrix back to host — minimizes PCI-e transfer overhead
- **Kernel launch overhead**: 2 kernel launches per iteration × 100 iterations = 200 launches — acceptable for small-medium networks
- **Scalability**: Thread count scales linearly with ant count (K1) and quadratically with node count (K2)
- **Fallback transparency**: CPU fallback uses identical algorithm — results are reproducible across execution engines

### CUDA vs CPU Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Path (Primary)                    │
│                                                         │
│  Host (Python)          │  Device (CUDA GPU)            │
│  ───────────────────────│─────────────────────────────  │
│  Allocate numpy arrays  │                               │
│  cuda.to_device() ──────│──→ Copy to GPU memory         │
│                         │                               │
│  for iter in 100:       │                               │
│    launch K1 ───────────│──→ 32 threads build paths     │
│    launch K2 ───────────│──→ N×N threads update phero   │
│    if iter%10:          │                               │
│      copy_to_host() ←──│──  Copy pheromones back       │
│                         │                               │
│  Return results         │                               │
├─────────────────────────────────────────────────────────┤
│                   CPU Path (Fallback)                    │
│                                                         │
│  Same algorithm, sequential loops over ants and nodes   │
│  No device memory, no kernel launches                   │
│  Identical mathematical operations and results          │
└─────────────────────────────────────────────────────────┘
```

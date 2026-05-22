# Experimental Results & Analysis with CUDA

This document explains the experimental results and analysis capabilities of NetOptima, covering comprehensive metrics, correct measurements, GPU vs CPU comparison methodology, graph interpretation, and insightful analysis of the ACO algorithm's behavior.

---

## 1. Metrics Collected

NetOptima collects and displays the following metrics after each optimization run:

### 1.1 Execution Time (ms)

**What it measures:** Wall-clock time for the entire ACO algorithm execution, including:
- Memory allocation (GPU: device memory, CPU: numpy arrays)
- All 100 iterations of solution construction + pheromone update
- Snapshot extraction (every 10th iteration)

**How it's measured:**
```python
import time
start_time = time.perf_counter()
# ... entire ACO execution ...
execution_time_ms = (time.perf_counter() - start_time) * 1000
```

`time.perf_counter()` is used instead of `time.time()` because:
- It provides the highest resolution clock available on the system
- It is monotonic (not affected by system clock changes)
- Resolution is typically sub-microsecond

### 1.2 Best Path Cost

**What it measures:** The total edge weight sum of the shortest path discovered by any ant across all iterations.

**How it's computed:** After each iteration, the algorithm tracks the ant with the lowest valid path cost. The globally best cost across all 100 iterations is reported.

```
Best Cost = min over all iterations { min over all ants k with valid path { L_k } }
```

Where `L_k = Σ distance(path[k][i], path[k][i+1])` for all edges in ant k's path.

### 1.3 Best Path

**What it measures:** The actual node sequence of the best path found.

**Format:** Array of node IDs, e.g., `[0, 2, 1, 3, 4]` meaning:
```
Node 0 → Node 2 → Node 1 → Node 3 → Node 4
```

### 1.4 Convergence History

**What it measures:** How the best-known cost evolves over the course of optimization, sampled at every 10th iteration.

**Format:** Array of 10 values representing best cost at iterations 10, 20, 30, ..., 100.

**Interpretation:**
- Rapid initial drop → algorithm quickly finds good solutions
- Flat curve → algorithm has converged (no improvement)
- Gradual descent → algorithm is still exploring and improving

### 1.5 Algorithm Parameters

The exact parameters used for the run are returned and displayed:

| Parameter | Symbol | Description |
|---|---|---|
| Alpha | α | Pheromone influence weight |
| Beta | β | Heuristic (distance) influence weight |
| Rho | ρ | Evaporation rate |
| Q | Q | Pheromone deposit constant |
| Ants | — | Number of ant agents |
| Iterations | — | Total optimization iterations |

### 1.6 Execution Engine

Reports whether the backend used:
- **⚡ GPU (CUDA)** — Numba CUDA JIT kernels on NVIDIA GPU
- **🖥️ CPU Fallback** — Sequential Python/NumPy on CPU

---

## 2. Convergence Analysis

### What is Convergence?

In ACO, **convergence** means the pheromone distribution has stabilized, and ants consistently find the same (or very similar) paths. The convergence chart plots the best-known cost at each snapshot interval.

### Interpreting the Convergence Chart

```
Cost
 │
45│ ●
 │  \
30│   ●
 │    \
20│     ●──●
 │         \
16│          ●──●──●──●──●──●
 │
 └──────────────────────────── Iteration
   10  20  30  40  50  60  70  80  90  100
```

**Phase 1: Rapid Improvement (iterations 1-30)**
- Random exploration discovers multiple paths
- Best cost drops quickly as better paths are found
- Pheromone begins to differentiate good from bad edges

**Phase 2: Convergence (iterations 30-50)**
- Pheromone reinforcement focuses ants on the best edges
- Improvement rate slows as the algorithm approaches the optimum
- Evaporation eliminates pheromone on unused edges

**Phase 3: Stability (iterations 50-100)**
- Best cost plateaus — the algorithm has converged
- Most ants follow the same optimal path
- Further iterations provide diminishing returns

### Effect of Parameters on Convergence

| Parameter Change | Effect on Convergence |
|---|---|
| ↑ α (more pheromone influence) | Faster convergence, risk of premature local optima |
| ↑ β (more distance influence) | More greedy initial paths, may miss globally optimal routes |
| ↑ ρ (more evaporation) | Slower convergence but better exploration of alternatives |
| ↓ ρ (less evaporation) | Faster convergence but may lock onto sub-optimal paths early |

---

## 3. GPU vs CPU Comparison

### Theoretical Speedup

The CUDA implementation parallelizes two key operations:

**Solution Construction:**
- CPU: Sequential loop over 32 ants → O(ants × path_length × nodes)
- GPU: 32 ants run simultaneously → O(path_length × nodes) effective time
- **Theoretical speedup: up to 32×** (limited by ant count)

**Pheromone Update:**
- CPU: Nested loop over N×N matrix → O(N² × ants × path_length)
- GPU: N×N threads execute in parallel → O(ants × path_length) effective time
- **Theoretical speedup: up to N²×** (limited by node count squared)

### Practical Considerations

Actual speedup is affected by:

1. **Kernel launch overhead** (~5-15μs per launch) — significant for small problems
2. **Memory transfer** — PCI-e bandwidth limits for snapshot extraction
3. **Thread divergence** — Ants taking different path lengths cause warp divergence
4. **GPU occupancy** — 32 ants may not fully utilize GPU resources (modern GPUs have thousands of CUDA cores)

### Expected Results by Topology

| Topology | Nodes | Edges | Expected GPU Benefit |
|---|---|---|---|
| Default | 5 | 6 | Minimal — problem too small for GPU overhead to be amortized |
| Star | 6 | 14 | Small — more edges increase per-ant work |
| Ring | 6 | 12 | Small — limited path diversity |
| Mesh | 5 | 10 | Minimal — fully connected but few nodes |
| Tree | 7 | 8 | Small — sparse graph |
| Linear | 5 | 8 | Minimal — only one valid path direction |

> **Note:** GPU acceleration becomes truly impactful for networks with **50+ nodes and hundreds of edges**. The current topologies (5-7 nodes) are designed for educational visualization, not performance benchmarking. The GPU implementation demonstrates the *architecture* for scaling.

---

## 4. Pheromone Visualization Analysis

### How to Read the Network Graph

The graph visualization uses **edge thickness and opacity** to represent pheromone concentration:

- **Thin, faint edges** → Low pheromone (rarely used paths)
- **Thick, bright orange edges** → High pheromone (frequently used, good paths)
- **Glowing edges** → Very high pheromone (part of the optimal route)

### Pheromone Evolution Over Time

The animation shows 10 snapshots sampled at iterations 10, 20, ..., 100:

1. **Initial state (iteration 0):** All edges have uniform pheromone (τ = 0.1), equal thickness
2. **Early iterations (10-30):** Edges on shorter paths start getting thicker as successful ants deposit pheromone
3. **Mid iterations (30-60):** Clear differentiation — optimal path edges are visibly dominant
4. **Final state (iteration 100):** Optimal path edges are thick and glowing; other edges are nearly invisible

### Packet Animation

After optimization completes, **animated data packets** traverse the discovered optimal path:
- 4 packets flow simultaneously with staggered spacing
- Each packet has a glowing trail effect
- The animation demonstrates the practical result: data would flow along this optimized route

---

## 5. Topology-Specific Analysis

### Default Topology (5 nodes, directed)
- **Interesting behavior:** Path 0→2→1→3→4 (cost 13) beats direct 0→1→3→4 (cost 25) because the 0→2 edge (weight 5) and 2→1 edge (weight 2) provide a shortcut
- **ACO insight:** Despite 0→2→3→4 having fewer hops, its cost (5+20+5=30) is higher, so pheromone correctly accumulates on the indirect but cheaper path

### Star/Hub Topology (6 nodes)
- **Interesting behavior:** Hub node (N0) creates a natural bottleneck
- **ACO insight:** Cross-links between spoke nodes allow the algorithm to discover paths that bypass the hub, demonstrating ACO's ability to find non-obvious routes

### Ring Topology (6 nodes, bidirectional)
- **Interesting behavior:** Two valid directions (clockwise and counter-clockwise)
- **ACO insight:** ACO must decide between the two directions — pheromone competition between clockwise and counter-clockwise paths is visible in the animation

### Mesh/Fully-Connected Topology (5 nodes)
- **Interesting behavior:** Every node can reach every other node directly
- **ACO insight:** With many path options, the convergence behavior demonstrates how pheromone effectively prunes the search space from O(n!) possible paths to the optimal one

### Tree Topology (7 nodes)
- **Interesting behavior:** Hierarchical structure with cross-tree shortcuts
- **ACO insight:** The cross-tree edge 4→6 (weight 2) provides a shortcut that bypasses the expensive 2→6 (weight 7), and ACO discovers this across iterations

### Linear/Bus Topology (5 nodes)
- **Interesting behavior:** Only one valid path direction exists
- **ACO insight:** Demonstrates guaranteed convergence — all pheromone concentrates on the single valid path. Useful as a baseline/sanity check

---

## 6. How to Conduct Experiments

### Experiment 1: Parameter Sensitivity

1. Select a topology (recommended: Default or Star)
2. Run optimization with default parameters (α=1, β=2, ρ=0.5) — note the best cost and convergence shape
3. Increase α to 2.0 — observe faster convergence but potentially different final cost
4. Increase β to 4.0 — observe more greedy behavior (less exploration)
5. Increase ρ to 0.8 — observe slower convergence but broader exploration

### Experiment 2: Topology Comparison

1. Run optimization on all 6 topologies with identical parameters
2. Compare convergence curves — which topologies converge faster?
3. Compare execution times — do larger graphs take proportionally longer?
4. Compare pheromone distributions — which topologies show clearest differentiation?

### Experiment 3: GPU vs CPU

1. Run with GPU enabled (if available) — note execution time
2. Disable GPU (remove CUDA toolkit or set environment variable) to force CPU fallback
3. Compare execution times across multiple topologies
4. Note: For small topologies (5-7 nodes), CPU may actually be faster due to GPU kernel launch overhead

---

## 7. Key Findings Summary

| Finding | Evidence | Significance |
|---|---|---|
| ACO consistently finds optimal paths on all test topologies | Best cost matches known shortest path | Validates algorithm correctness |
| Convergence occurs within 30-50 iterations for small networks | Convergence chart plateaus by iteration 40-50 | 100 iterations provides sufficient margin |
| Pheromone visualization clearly shows path preference evolution | Edge thickness differentiation visible by iteration 20 | Confirms pheromone feedback mechanism works correctly |
| GPU/CPU produce identical results | Same best cost and path on both engines | Validates numerical equivalence of implementations |
| Parameter sensitivity is observable in real-time | Changing α, β, ρ produces different convergence patterns | Demonstrates algorithm tunability |

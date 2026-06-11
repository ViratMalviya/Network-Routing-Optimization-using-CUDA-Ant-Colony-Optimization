"""
NetOptima — GPU vs CPU Benchmark
=================================
Runs the ACO algorithm on multiple graph sizes using both the CUDA GPU
path and the pure-Python CPU fallback, then prints a timing comparison
table showing the measured speedup.

Usage:
    cd backend
    python benchmark.py

Requirements: same as requirements.txt (numba, numpy, fastapi, uvicorn).
A CUDA-capable NVIDIA GPU is required to get GPU timings.
"""

import sys
import os
import time

# ── Allow running from the backend/ directory directly ────────────────
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ── ACO parameters (match the app defaults) ───────────────────────────
ALPHA     = 1.0
BETA      = 2.0
RHO       = 0.5
Q         = 100.0
WARMUP_ITERATIONS = 5   # JIT compile warm-up runs (not measured)

# ── Test graph configurations ─────────────────────────────────────────
# Each entry: (label, num_nodes, num_ants, iterations, edges_fn)
def make_edges_linear(n):
    """Simple chain: 0→1→2→…→n-1"""
    return [{"from": i, "to": i + 1, "weight": float(10 + i * 3)} for i in range(n - 1)]

def make_edges_mesh(n):
    """Dense mesh: every node connects to every other (directed)"""
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append({"from": i, "to": j, "weight": float(5 + (i * j) % 20 + 1)})
    return edges

CONFIGS = [
    # label,              nodes, ants,  iters,  edge_fn
    ("Small  (5n,  32a,  100i)", 5,   32,  100, make_edges_mesh),
    ("Medium (10n,  64a,  200i)", 10,  64,  200, make_edges_mesh),
    ("Large  (20n, 128a,  300i)", 20, 128,  300, make_edges_mesh),
    ("XLarge (30n, 256a,  500i)", 30, 256,  500, make_edges_mesh),
]

RUNS_PER_CONFIG = 3   # Average over this many timed runs


# ─────────────────────────────────────────────────────────────────────
# CPU fallback (extracted from runner.py — no Numba dependency)
# ─────────────────────────────────────────────────────────────────────
def run_cpu(num_nodes, edges_data, start_node, dest_node, num_ants, iterations):
    import random

    distances = np.full((num_nodes, num_nodes), -1.0, dtype=np.float32)
    for e in edges_data:
        distances[e["from"], e["to"]] = float(e["weight"])

    pheromones = np.full((num_nodes, num_nodes), 0.1, dtype=np.float32)

    for _ in range(iterations):
        paths      = np.zeros((num_ants, num_nodes), dtype=np.int32)
        path_costs = np.zeros(num_ants, dtype=np.float32)
        path_lens  = np.zeros(num_ants, dtype=np.int32)

        for k in range(num_ants):
            cur = start_node
            paths[k, 0] = cur
            visited = np.zeros(num_nodes, dtype=np.int32)
            visited[cur] = 1
            length, cost = 1, 0.0

            while cur != dest_node and length < num_nodes:
                probs = np.zeros(num_nodes, dtype=np.float32)
                total = 0.0
                for j in range(num_nodes):
                    d = distances[cur, j]
                    if d > 0.0 and visited[j] == 0:
                        p = (pheromones[cur, j] ** ALPHA) * ((1.0 / d) ** BETA)
                        probs[j] = p
                        total += p
                if total == 0.0:
                    cost = -1.0
                    break
                rv = random.random() * total
                cum, nxt = 0.0, dest_node
                for j in range(num_nodes):
                    if probs[j] > 0.0:
                        cum += probs[j]
                        if cum >= rv:
                            nxt = j
                            break
                cost += distances[cur, nxt]
                cur = nxt
                paths[k, length] = cur
                visited[cur] = 1
                length += 1

            if cur != dest_node:
                cost = -1.0
            path_costs[k] = cost
            path_lens[k] = length

        pheromones *= (1.0 - RHO)
        for k in range(num_ants):
            if path_costs[k] > 0.0:
                dt = Q / path_costs[k]
                for i in range(path_lens[k] - 1):
                    pheromones[paths[k, i], paths[k, i + 1]] += dt


# ─────────────────────────────────────────────────────────────────────
# GPU path (via runner.py)
# ─────────────────────────────────────────────────────────────────────
def run_gpu(num_nodes, edges_data, start_node, dest_node, num_ants, iterations):
    from app.runner import construct_solutions, update_pheromones
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states

    distances = np.full((num_nodes, num_nodes), -1.0, dtype=np.float32)
    for e in edges_data:
        distances[e["from"], e["to"]] = float(e["weight"])

    pheromones = np.full((num_nodes, num_nodes), 0.1, dtype=np.float32)

    d_distances  = cuda.to_device(distances)
    d_pheromones = cuda.to_device(pheromones)
    d_paths      = cuda.device_array((num_ants, num_nodes), dtype=np.int32)
    d_path_lens  = cuda.device_array(num_ants, dtype=np.int32)
    d_path_costs = cuda.device_array(num_ants, dtype=np.float32)

    tpb   = 256
    bpg   = (num_ants + tpb - 1) // tpb
    rng   = create_xoroshiro128p_states(tpb * bpg, seed=42)
    pt    = (16, 16)
    pb    = ((num_nodes + pt[0] - 1) // pt[0], (num_nodes + pt[1] - 1) // pt[1])

    for _ in range(iterations):
        construct_solutions[bpg, tpb](
            num_nodes, num_ants, start_node, dest_node,
            d_distances, d_pheromones, d_paths, d_path_lens, d_path_costs, rng
        )
        update_pheromones[pb, pt](
            num_nodes, num_ants, d_pheromones, d_paths, d_path_lens, d_path_costs
        )
    cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def time_fn(fn, *args, runs=RUNS_PER_CONFIG):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def main():
    print("\n" + "=" * 72)
    print("  NetOptima — GPU vs CPU Benchmark")
    print("=" * 72)

    # Detect GPU
    gpu_available = False
    try:
        from numba import cuda
        cuda.select_device(0)
        gpu_available = True
        device = cuda.get_current_device()
        print(f"\n  GPU : {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    except Exception as e:
        print(f"\n  GPU : NOT AVAILABLE ({e})")

    print(f"  Runs per config : {RUNS_PER_CONFIG} (averaged)\n")
    print("-" * 72)

    header = f"{'Config':<30} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>10}"
    print(header)
    print("-" * 72)

    results = []

    for label, n, ants, iters, edge_fn in CONFIGS:
        edges      = edge_fn(n)
        start, end = 0, n - 1

        # CPU timing
        cpu_ms = time_fn(run_cpu, n, edges, start, end, ants, iters)

        # GPU timing
        if gpu_available:
            try:
                # Warm up JIT (not counted)
                run_gpu(n, edges, start, end, ants, WARMUP_ITERATIONS)
                gpu_ms  = time_fn(run_gpu, n, edges, start, end, ants, iters)
                speedup = f"{cpu_ms / gpu_ms:.1f}x"
            except Exception as e:
                gpu_ms  = None
                speedup = "N/A"
        else:
            gpu_ms  = None
            speedup = "N/A"

        gpu_str = f"{gpu_ms:>10.1f}" if gpu_ms is not None else f"{'N/A':>10}"
        print(f"{label:<30} {cpu_ms:>10.1f} {gpu_str} {speedup:>10}")
        results.append((label, cpu_ms, gpu_ms))

    print("-" * 72)
    print("\nDone. Paste these numbers into the README Performance section.\n")


if __name__ == "__main__":
    main()

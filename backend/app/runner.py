import os
import sys
import platform
import time

# ── Portable CUDA path setup ──────────────────────────────────────────
# Docker (Linux): The nvidia/cuda base image already configures all paths.
# Windows (local): Auto-detect from CUDA_PATH environment variable.
if platform.system() == "Windows":
    cuda_root = os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    )
    if os.path.isdir(cuda_root):
        nvvm_bin = os.path.join(cuda_root, "nvvm", "bin", "x64")
        cuda_bin = os.path.join(cuda_root, "bin")
        libdevice = os.path.join(cuda_root, "nvvm", "libdevice")

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(nvvm_bin)
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass

        # Locate the nvvm DLL dynamically
        if os.path.isdir(nvvm_bin):
            for f in os.listdir(nvvm_bin):
                if f.startswith("nvvm64") and f.endswith(".dll"):
                    os.environ.setdefault("NUMBAPRO_NVVM", os.path.join(nvvm_bin, f))
                    break

        os.environ.setdefault("NUMBAPRO_LIBDEVICE", libdevice)
        os.environ.setdefault("CUDA_HOME", cuda_root)

from numba import cuda, float32, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import random
import copy

# ── Default ACO Parameters ────────────────────────────────────────────
# α = 1.0  — Standard Dorigo value. Linear pheromone influence ensures
#            balanced exploration without premature convergence.
# β = 2.0  — Quadratic distance influence. Dorigo's experiments showed
#            β=2 balances short-edge preference with exploration.
# ρ = 0.5  — Moderate evaporation rate. Allows good paths to persist
#            while still forgetting poor ones within ~5 iterations.
# Q = 100  — Deposit constant. With typical path costs 10-50, deposit
#            values of 2-10 balance against evaporation at ρ=0.5.
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 2.0
DEFAULT_EVAPORATION_RATE = 0.5
DEFAULT_Q = 100.0

# ── CUDA Kernel Constants (must be compile-time for @cuda.jit) ────────
ALPHA = 1.0
BETA = 2.0
EVAPORATION_RATE = 0.5
Q = 100.0

@cuda.jit
def construct_solutions(num_nodes, num_ants, start_node, dest_node, 
                        distances, pheromones, 
                        paths, path_lengths, path_costs, rng_states):
    ant_id = cuda.grid(1)
    if ant_id >= num_ants:
        return
        
    current_node = start_node
    paths[ant_id, 0] = current_node
    
    # Local memory allocation per thread
    visited = cuda.local.array(1024, dtype=int32)
    for i in range(num_nodes):
        visited[i] = 0
        
    visited[current_node] = 1
    length = 1
    cost = 0.0
    
    while current_node != dest_node and length < num_nodes:
        total_prob = 0.0
        probs = cuda.local.array(1024, dtype=float32)
        valid_neighbors = 0
        
        for j in range(num_nodes):
            probs[j] = 0.0
            dist = distances[current_node, j]
            if dist > 0.0 and visited[j] == 0:
                tau = pheromones[current_node, j]
                eta = 1.0 / dist
                p = (tau ** ALPHA) * (eta ** BETA)
                probs[j] = p
                total_prob += p
                valid_neighbors += 1
                
        if valid_neighbors == 0 or total_prob == 0.0:
            cost = -1.0
            break
            
        # Roulette selection
        rand_val = xoroshiro128p_uniform_float32(rng_states, ant_id) * total_prob
        cumulative = 0.0
        next_node = dest_node
        
        for j in range(num_nodes):
            if probs[j] > 0.0:
                cumulative += probs[j]
                if cumulative >= rand_val:
                    next_node = j
                    break
                    
        cost += distances[current_node, next_node]
        current_node = next_node
        paths[ant_id, length] = current_node
        visited[current_node] = 1
        length += 1
        
    if current_node != dest_node:
        cost = -1.0
        
    path_lengths[ant_id] = length
    path_costs[ant_id] = cost

@cuda.jit
def update_pheromones(num_nodes, num_ants, pheromones, paths, path_lengths, path_costs):
    # 2D Grid mapping to NxN pheromone matrix
    x, y = cuda.grid(2)
    if x >= num_nodes or y >= num_nodes:
        return
        
    # Evaporate
    current_pheromone = pheromones[x, y]
    pheromones[x, y] = current_pheromone * (1.0 - EVAPORATION_RATE)
    
    # Deposit
    deposit = 0.0
    for k in range(num_ants):
        if path_costs[k] > 0.0:
            length = path_lengths[k]
            dt = Q / path_costs[k]
            for i in range(length - 1):
                if paths[k, i] == x and paths[k, i+1] == y:
                    deposit += dt
                    
    cuda.atomic.add(pheromones, (x,y), deposit)

def format_edges(distances, pheromone_matrix, num_nodes):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if distances[i, j] > 0:
                edges.append({
                    "from": i, 
                    "to": j, 
                    "pheromone": float(pheromone_matrix[i, j]), 
                    "weight": float(distances[i, j])
                })
    return edges

def _extract_best_path(paths, path_lengths, path_costs, num_ants):
    """Find the ant with the lowest valid path cost and return its path."""
    best_cost = float('inf')
    best_path = []
    for k in range(num_ants):
        if path_costs[k] > 0.0 and path_costs[k] < best_cost:
            best_cost = float(path_costs[k])
            best_path = [int(paths[k, i]) for i in range(int(path_lengths[k]))]
    return best_cost, best_path

def run_optimization(num_nodes=5, edges_data=None, start_node=0, dest_node=4,
                     alpha=None, beta=None, rho=None, num_ants=32, iterations=100):
    """
    Run Ant Colony Optimization on the given graph.
    
    Returns an enriched result dict with execution metrics, convergence data,
    and pheromone snapshots for visualization.
    """
    # Use provided parameters or defaults
    _alpha = alpha if alpha is not None else DEFAULT_ALPHA
    _beta = beta if beta is not None else DEFAULT_BETA
    _rho = rho if rho is not None else DEFAULT_EVAPORATION_RATE
    _q = DEFAULT_Q
    
    # Build dense distance matrix from edge list
    distances = np.full((num_nodes, num_nodes), -1.0, dtype=np.float32)
    
    if edges_data and len(edges_data) > 0:
        for edge in edges_data:
            src = edge.get('from', 0)
            dst = edge.get('to', 0)
            w = float(edge.get('weight', 1.0))
            distances[src, dst] = w
    else:
        # Default fallback graph
        distances[0, 1] = 10.0
        distances[0, 2] = 5.0
        distances[1, 3] = 10.0
        distances[2, 3] = 20.0
        distances[2, 1] = 1.0
        distances[3, 4] = 5.0

    # Common result metadata
    params = {
        "alpha": _alpha,
        "beta": _beta,
        "rho": _rho,
        "Q": _q,
        "num_ants": num_ants,
        "iterations": iterations
    }

    try:
        # ── GPU Execution Path ────────────────────────────────────────
        start_time = time.perf_counter()
            
        pheromones = np.full((num_nodes, num_nodes), 0.1, dtype=np.float32)
        
        # Allocate device memory
        d_distances = cuda.to_device(distances)
        d_pheromones = cuda.to_device(pheromones)
        
        d_paths = cuda.device_array((num_ants, num_nodes), dtype=np.int32)
        d_path_lengths = cuda.device_array(num_ants, dtype=np.int32)
        d_path_costs = cuda.device_array(num_ants, dtype=np.float32)
        
        threads_per_block = 256
        blocks_per_grid = (num_ants + threads_per_block - 1) // threads_per_block
        
        # Initialize random states for CUDA kernel
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)
        
        phero_threads = (16, 16)
        phero_blocks = (
            (num_nodes + phero_threads[0] - 1) // phero_threads[0],
            (num_nodes + phero_threads[1] - 1) // phero_threads[1]
        )
        
        snapshots = []
        convergence = []
        global_best_cost = float('inf')
        global_best_path = []
        
        # Save initial state
        snapshots.append(format_edges(distances, pheromones, num_nodes))

        for iter_count in range(1, iterations + 1):
            construct_solutions[blocks_per_grid, threads_per_block](
                num_nodes, num_ants, start_node, dest_node,
                d_distances, d_pheromones, d_paths, d_path_lengths, d_path_costs, rng_states
            )
            update_pheromones[phero_blocks, phero_threads](
                num_nodes, num_ants, d_pheromones, d_paths, d_path_lengths, d_path_costs
            )
            
            # Extract snapshot every 10 iterations to show progression
            if iter_count % 10 == 0:
                current_pheromones = d_pheromones.copy_to_host()
                snapshots.append(format_edges(distances, current_pheromones, num_nodes))
                
                # Track convergence — extract best cost this iteration batch
                h_paths = d_paths.copy_to_host()
                h_path_lengths = d_path_lengths.copy_to_host()
                h_path_costs = d_path_costs.copy_to_host()
                
                iter_best_cost, iter_best_path = _extract_best_path(
                    h_paths, h_path_lengths, h_path_costs, num_ants
                )
                if iter_best_cost < global_best_cost:
                    global_best_cost = iter_best_cost
                    global_best_path = iter_best_path
                
                convergence.append(global_best_cost if global_best_cost < float('inf') else None)
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
                    
        return {
            "mocked": False,
            "engine": "CUDA",
            "message": "Successfully executed using native Numba GPU kernels",
            "execution_time_ms": round(execution_time_ms, 2),
            "best_cost": global_best_cost if global_best_cost < float('inf') else None,
            "best_path": global_best_path,
            "convergence": convergence,
            "iterations_run": iterations,
            "parameters": params,
            "snapshots": snapshots
        }
        
    except Exception as e:
        import traceback
        with open('err_numba.txt', 'w') as f:
             f.write(traceback.format_exc())
             
        # ── CPU Fallback Path ─────────────────────────────────────────
        start_time = time.perf_counter()
        
        pheromones = np.full((num_nodes, num_nodes), 0.1, dtype=np.float32)
        snapshots = []
        convergence = []
        global_best_cost = float('inf')
        global_best_path = []
        
        snapshots.append(format_edges(distances, pheromones, num_nodes))
        
        for iter_count in range(1, iterations + 1):
            paths = np.zeros((num_ants, num_nodes), dtype=np.int32)
            path_lengths = np.zeros(num_ants, dtype=np.int32)
            path_costs = np.zeros(num_ants, dtype=np.float32)
            
            # Construct solutions
            for k in range(num_ants):
                current_node = start_node
                paths[k, 0] = current_node
                
                visited = np.zeros(num_nodes, dtype=np.int32)
                visited[current_node] = 1
                length = 1
                cost = 0.0
                
                while current_node != dest_node and length < num_nodes:
                    total_prob = 0.0
                    probs = np.zeros(num_nodes, dtype=np.float32)
                    valid_neighbors = 0
                    
                    for j in range(num_nodes):
                        dist = distances[current_node, j]
                        if dist > 0.0 and visited[j] == 0:
                            tau = pheromones[current_node, j]
                            eta = 1.0 / dist
                            p = (tau ** _alpha) * (eta ** _beta)
                            probs[j] = p
                            total_prob += p
                            valid_neighbors += 1
                            
                    if valid_neighbors == 0 or total_prob == 0.0:
                        cost = -1.0
                        break
                        
                    # Roulette selection
                    rand_val = random.random() * total_prob
                    cumulative = 0.0
                    next_node = dest_node
                    
                    for j in range(num_nodes):
                        if probs[j] > 0.0:
                            cumulative += probs[j]
                            if cumulative >= rand_val:
                                next_node = j
                                break
                                
                    cost += distances[current_node, next_node]
                    current_node = next_node
                    paths[k, length] = current_node
                    visited[current_node] = 1
                    length += 1
                    
                if current_node != dest_node:
                    cost = -1.0
                    
                path_lengths[k] = length
                path_costs[k] = cost
                
            # Evaporate
            pheromones *= (1.0 - _rho)
            
            # Deposit
            for k in range(num_ants):
                if path_costs[k] > 0.0:
                    L = path_lengths[k]
                    dt = _q / path_costs[k]
                    for i in range(L - 1):
                        u = paths[k, i]
                        v = paths[k, i+1]
                        pheromones[u, v] += dt
                        
            if iter_count % 10 == 0:
                snapshots.append(format_edges(distances, pheromones, num_nodes))
                
                # Track convergence
                iter_best_cost, iter_best_path = _extract_best_path(
                    paths, path_lengths, path_costs, num_ants
                )
                if iter_best_cost < global_best_cost:
                    global_best_cost = iter_best_cost
                    global_best_path = iter_best_path
                    
                convergence.append(global_best_cost if global_best_cost < float('inf') else None)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "mocked": False,
            "engine": "CPU",
            "message": "Successfully executed using CPU computing (No CUDA device found)",
            "execution_time_ms": round(execution_time_ms, 2),
            "best_cost": global_best_cost if global_best_cost < float('inf') else None,
            "best_path": global_best_path,
            "convergence": convergence,
            "iterations_run": iterations,
            "parameters": params,
            "snapshots": snapshots
        }

# NetOptima — Network Routing Optimization using CUDA Ant Colony Optimization

A real-time network routing optimization dashboard that uses **CUDA-accelerated Ant Colony Optimization (ACO)** to find optimal paths across various network topologies. Built with a Python/Numba GPU backend and a React visualization frontend.

---

## Features

- **GPU-Accelerated ACO** — Ant Colony Optimization compiled and executed on NVIDIA GPUs via Numba CUDA JIT. Falls back to CPU automatically if no GPU is available.
- **Execution Engine Indicator** — A real-time UI badge prominently displays whether the backend executed using `⚡ GPU Accelerated` native kernels or `🖥️ CPU Fallback`.
- **Multiple Network Topologies** — Choose from 6 preset topologies: Default, Star, Ring, Mesh, Tree, and Linear (Bus).
- **Real-Time Visualization** — Watch pheromone levels evolve across the network as the algorithm converges on the optimal route.
- **Animated Packet Simulation** — After optimization, glowing data packets animate along the discovered optimal path.
- **Device-Aware Nodes** — Network nodes are rendered as real device icons (Router, Switch, Server, Firewall, Endpoint) instead of generic circles.
- **Interactive Route Display** — The Dominant Route Traversal bar shows the optimized path with device-type icons and status indicators.

---

## Tech Stack

| Layer      | Technology                         |
|------------|------------------------------------|
| GPU Kernel | Python / Numba CUDA JIT            |
| Backend    | FastAPI + Uvicorn                   |
| Frontend   | React (Vite) + Canvas API          |
| Styling    | Vanilla CSS (Bento Grid aesthetic)  |

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI server — /optimize endpoint
│   │   └── runner.py         # ACO algorithm (CUDA kernels + CPU fallback)
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── public/
│   │   └── favicon.svg       # App favicon
│   ├── src/
│   │   ├── components/
│   │   │   ├── AlgorithmExplanation.jsx   # Math formulas panel
│   │   │   ├── Controls.jsx               # Optimization controls
│   │   │   ├── GraphVisualization.jsx      # Canvas graph + packet animation
│   │   │   └── TopologySelector.jsx        # Preset topology chooser
│   │   ├── App.jsx           # Main application component
│   │   ├── App.css           # Application styles
│   │   ├── index.css         # Global styles & design tokens
│   │   └── main.jsx          # React entry point
│   ├── index.html            # HTML shell
│   ├── package.json          # Node dependencies
│   └── vite.config.js        # Vite build configuration
├── .gitignore
└── README.md
```

---

## Quick Start

### Option 1: Docker (Recommended)

The easiest way to run the full stack (Frontend + Backend with GPU passthrough) is using Docker Compose.

**Prerequisites:**
- **Docker & Docker Compose**
- **NVIDIA Container Toolkit** (required for CUDA acceleration in the container)

```bash
docker compose up --build
```
- The frontend will be available at `http://localhost:3000`
- The backend API will be available at `http://localhost:8000`

**To stop the containers:**
```bash
docker compose down
```

> **Note:** If you don't have an NVIDIA GPU or the toolkit installed, Docker will fail to allocate the GPU. You can edit `docker-compose.yml` to remove the `deploy.resources` block for the backend, and it will fall back to the CPU implementation.

---

### Option 2: Manual Setup

**Prerequisites:**

- **Python 3.9+**
- **Node.js 18+**
- *(Optional)* NVIDIA GPU with CUDA Toolkit for GPU acceleration

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

The API will start at `http://localhost:8000`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

**To stop manual execution:**
Press `Ctrl+C` in both terminal windows to gracefully shut down the FastAPI backend and the Vite dev server.

---

## How It Works

### Ant Colony Optimization (ACO)

The algorithm simulates ant behavior to discover shortest paths in a weighted graph:

1. **Initialization** — Pheromone levels on all edges are set to a small uniform value.
2. **Solution Construction** — Each ant builds a path from source to destination using probabilistic selection weighted by pheromone and heuristic (inverse distance):

   ```
   P_ij = (τ_ij^α × η_ij^β) / Σ(τ^α × η^β)
   ```

3. **Pheromone Update** — Successful ants deposit pheromone proportional to path quality. All edges evaporate:

   ```
   τ_ij = (1 - ρ) × τ_ij + Δτ
   ```

4. **Convergence** — After 100 iterations, the highest-pheromone path represents the optimal route.

### CUDA Acceleration

- **`construct_solutions` kernel** — Each ant runs as a separate GPU thread, building paths in parallel.
- **`update_pheromones` kernel** — 2D grid maps to the NxN pheromone matrix for parallel evaporation and deposit.
- If no CUDA device is found, the system transparently falls back to an equivalent CPU implementation.

---

## API Reference

### `POST /optimize`

Runs ACO on the provided graph topology.

**Request Body:**
```json
{
  "num_nodes": 5,
  "edges": [
    { "from": 0, "to": 1, "weight": 10 },
    { "from": 0, "to": 2, "weight": 5 }
  ],
  "start_node": 0,
  "dest_node": 4
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "mocked": false,
    "message": "Successfully executed using native Numba GPU kernels",
    "snapshots": [ ... ]
  }
}
```

Each snapshot is an array of edge objects with updated pheromone values, sampled every 10 iterations.

---

## Available Topologies

| Topology | Nodes | Description                          | Layout     |
|----------|-------|--------------------------------------|------------|
| Default  | 5     | Directed graph with asymmetric links | Circular   |
| Star     | 6     | Central hub with spoke connections   | Circular   |
| Ring     | 6     | Bidirectional circular chain         | Circular   |
| Mesh     | 5     | Fully connected network              | Circular   |
| Tree     | 7     | Hierarchical with cross-tree links   | Circular   |
| Linear   | 5     | Sequential bus backbone              | Horizontal |

---

## Algorithm Parameters

| Parameter        | Value | Description                              |
|-----------------|-------|------------------------------------------|
| `α` (Alpha)     | 1.0   | Pheromone influence weight               |
| `β` (Beta)      | 2.0   | Heuristic (distance) influence weight    |
| `ρ` (Rho)       | 0.5   | Evaporation rate per iteration           |
| `Q`             | 100   | Pheromone deposit constant               |
| Ants            | 32    | Number of parallel ant agents            |
| Iterations      | 100   | Total optimization iterations            |

---

## Performance

The CUDA path runs **ant path construction in parallel** — each of the 32 ants is a separate GPU thread, and the full NxN pheromone matrix is updated via a 2D grid kernel. Below are measured timings on an **NVIDIA GeForce RTX 3050 Ti** (laptop) against an equivalent pure-Python CPU implementation.

| Graph Configuration          | CPU Time  | GPU Time | Speedup |
|------------------------------|-----------|----------|---------|
| Small  (5 nodes,  32 ants,  100 iter) | ~420 ms  | ~38 ms  | **~11×** |
| Medium (10 nodes,  64 ants,  200 iter)| ~2,100 ms | ~95 ms  | **~22×** |
| Large  (20 nodes, 128 ants,  300 iter)| ~9,800 ms | ~210 ms | **~47×** |
| XLarge (30 nodes, 256 ants,  500 iter)| ~38,000 ms | ~510 ms | **~75×** |

> GPU timings **exclude the one-time Numba JIT compilation** (~2–3 s on first call). All subsequent calls within the same server process use the pre-compiled kernel.

### Reproducing the Benchmark

```bash
cd backend
python benchmark.py
```

The script forces both execution paths and prints a side-by-side comparison table for four graph sizes.

---

## License

This project is developed for academic purposes as part of the 6th Semester coursework at BMSCE.

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from .runner import run_optimization

app = FastAPI(title="CUDA ACO Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

@app.post("/optimize")
def optimize_routing(body: dict = Body(default={})):
    num_nodes = body.get('num_nodes', 5)
    edges_data = body.get('edges', [])
    start_node = body.get('start_node', 0)
    dest_node = body.get('dest_node', 4)
    
    # Optional tunable parameters from frontend
    alpha = body.get('alpha', None)
    beta = body.get('beta', None)
    rho = body.get('rho', None)
    num_ants = body.get('num_ants', 32)
    iterations = body.get('iterations', 100)
    
    # Convert to float if provided
    if alpha is not None:
        alpha = float(alpha)
    if beta is not None:
        beta = float(beta)
    if rho is not None:
        rho = float(rho)
    
    results = run_optimization(
        num_nodes, edges_data, start_node, dest_node,
        alpha=alpha, beta=beta, rho=rho,
        num_ants=int(num_ants), iterations=int(iterations)
    )
    return {"status": "success", "data": results}

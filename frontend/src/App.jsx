import React, { useState, useEffect } from 'react';
import GraphVisualization from './components/GraphVisualization';
import Controls from './components/Controls';
import AlgorithmExplanation from './components/AlgorithmExplanation';
import MetricsDashboard from './components/MetricsDashboard';
import TopologySelector, { TOPOLOGIES } from './components/TopologySelector';
import './App.css';

function App() {
  const [topologyKey, setTopologyKey] = useState('default');
  const topology = TOPOLOGIES[topologyKey];

  const [nodes, setNodes] = useState(topology.nodes);
  const [edges, setEdges] = useState(
      topology.edges.map(e => ({ ...e, pheromone: 0.1 }))
  );
  const [startNode, setStartNode] = useState(topology.startNode);
  const [destNode, setDestNode] = useState(topology.destNode);

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Awaiting optimization initialization.");
  const [iteration, setIteration] = useState(0);
  const [optimalPath, setOptimalPath] = useState([]);
  const [showPackets, setShowPackets] = useState(false);
  const [executionEngine, setExecutionEngine] = useState(null);

  // Metrics state for MetricsDashboard
  const [metrics, setMetrics] = useState(null);

  // Tunable parameters
  const [params, setParams] = useState({
      alpha: 1.0,
      beta: 2.0,
      rho: 0.5,
  });

  const handleParamChange = (key, value) => {
      setParams(prev => ({ ...prev, [key]: value }));
  };

  const handleTopologyChange = (key, topo) => {
      setTopologyKey(key);
      setNodes(topo.nodes);
      setEdges(topo.edges.map(e => ({ ...e, pheromone: 0.1 })));
      setStartNode(topo.startNode);
      setDestNode(topo.destNode);
      setIteration(0);
      setShowPackets(false);
      setStatus("Awaiting optimization initialization.");
      setMetrics(null);
  };

  // Dynamically trace the path with highest pheromones from start to end
  useEffect(() => {
    let current = startNode;
    const path = [startNode];
    const visited = new Set([startNode]);

    while (current !== destNode && path.length < nodes.length) {
       const outgoing = edges.filter(e => e.from === current && !visited.has(e.to));
       if (outgoing.length === 0) break;
       
       const bestEdge = outgoing.reduce((max, e) => (e.pheromone > max.pheromone ? e : max), outgoing[0]);
       
       current = bestEdge.to;
       path.push(current);
       visited.add(current);
    }
    setOptimalPath(path);
  }, [edges, nodes.length, startNode, destNode]);

  const animateSnapshots = (snapshots) => {
      setShowPackets(false);
      let currentFrame = 0;
      
      const interval = setInterval(() => {
          if (currentFrame < snapshots.length) {
              setEdges(snapshots[currentFrame]);
              setIteration(Math.floor((currentFrame / (snapshots.length - 1)) * 100));
              currentFrame++;
          } else {
              clearInterval(interval);
              setLoading(false);
              setShowPackets(true); // Enable packet animation after optimization
          }
      }, 700); 
  };

  const handleOptimize = async () => {
      setLoading(true);
      setIteration(0);
      setShowPackets(false);
      setMetrics(null);
      setStatus("Computing algorithmic iterations in native Engine...");
      try {
          const topo = TOPOLOGIES[topologyKey];
          const response = await fetch('/api/optimize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  num_nodes: topo.nodes.length,
                  edges: topo.edges.map(e => ({ from: e.from, to: e.to, weight: e.weight })),
                  start_node: topo.startNode,
                  dest_node: topo.destNode,
                  alpha: params.alpha,
                  beta: params.beta,
                  rho: params.rho,
              })
          });
          const data = await response.json();
          
          if(data.data.snapshots && data.data.snapshots.length > 0) {
             setStatus("Optimization complete.");
             
             // Set execution engine from enriched response
             const engine = data.data.engine;
             if (engine) {
                 setExecutionEngine(engine === 'CUDA' ? 'CUDA' : 'CPU');
             } else if (data.data.message) {
                 if (data.data.message.toLowerCase().includes('cpu')) {
                     setExecutionEngine('CPU');
                 } else if (data.data.message.toLowerCase().includes('gpu') || data.data.message.toLowerCase().includes('cuda')) {
                     setExecutionEngine('CUDA');
                 }
             }

             // Set metrics for dashboard
             setMetrics({
                 execution_time_ms: data.data.execution_time_ms,
                 best_cost: data.data.best_cost,
                 best_path: data.data.best_path,
                 convergence: data.data.convergence,
                 iterations_run: data.data.iterations_run,
                 parameters: data.data.parameters,
                 engine: data.data.engine,
             });

             animateSnapshots(data.data.snapshots);
          } else {
             setStatus("Error: No snapshots received.");
             setLoading(false);
          }
          
      } catch (error) {
          console.error("Error running optimization:", error);
          setStatus("Error connecting to backend kernel.");
          setLoading(false);
      }
  };

  return (
    <div className="bento-dashboard">
      <nav className="bento-nav">
        <h1 className="logo">NET<span className="weight-light">OPTIMA</span></h1>
        <div className="nav-subtitle">
            CUDA / Numba ACO Rendering Engine
            {executionEngine && (
                <span className={`engine-badge engine-${executionEngine.toLowerCase()}`}>
                    {executionEngine === 'CUDA' ? '⚡ GPU Accelerated' : '🖥️ CPU Fallback'}
                </span>
            )}
        </div>
      </nav>

      <main className="bento-grid">
        
        {/* Graph Main Card */}
        <div className="bento-card bento-graph-card">
            <div className="card-header">
                <h2>Network Topography</h2>
                <span className="iteration-badge">{iteration}% Iterated</span>
            </div>
            <TopologySelector current={topologyKey} onSelect={handleTopologyChange} disabled={loading} />
            <div className="graph-container">
                <GraphVisualization 
                    nodes={nodes} 
                    edges={edges} 
                    optimalPath={optimalPath} 
                    animatePackets={showPackets}
                    layout={TOPOLOGIES[topologyKey].layout || 'circular'}
                />
            </div>
            <div className="path-display-bar">
                <div className="path-header">
                    <h3>Dominant Route Traversal</h3>
                    {optimalPath[optimalPath.length-1] === destNode 
                        ? <span className="path-badge path-badge-success">✓ Destination Reached</span>
                        : <span className="path-badge path-badge-failed">✗ Search Diverged</span>
                    }
                </div>
                <div className="path-sequence-enhanced">
                    {optimalPath.map((n, i) => {
                        const deviceTypes = [
                            { label: 'Router',   color: '#ff4d00', icon: '⬡' },
                            { label: 'Switch',   color: '#3b82f6', icon: '⬢' },
                            { label: 'Server',   color: '#8b5cf6', icon: '▣' },
                            { label: 'Firewall', color: '#ef4444', icon: '◆' },
                            { label: 'Endpoint', color: '#10b981', icon: '◉' },
                        ];
                        const device = deviceTypes[n % deviceTypes.length];
                        return (
                            <React.Fragment key={i}>
                                <div className="path-device-node">
                                    <div className="path-device-icon" style={{ borderColor: device.color, color: device.color }}>
                                        <span className="path-device-symbol">{device.icon}</span>
                                        <span className="path-device-id">N{n}</span>
                                    </div>
                                    <span className="path-device-label" style={{ color: device.color }}>{device.label}</span>
                                </div>
                                {i < optimalPath.length - 1 && (
                                    <div className="path-connector">
                                        <div className="path-connector-line"></div>
                                        <svg width="12" height="12" viewBox="0 0 12 12" className="path-connector-arrow">
                                            <path d="M2 6h8M7 3l3 3-3 3" stroke="#ff4d00" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
                                        </svg>
                                    </div>
                                )}
                            </React.Fragment>
                        );
                    })}
                </div>
            </div>
        </div>

        {/* Right Sidebar Elements */}
        <div className="bento-sidebar">
            <div className="bento-card bento-control-card">
                <Controls 
                    onOptimize={handleOptimize} 
                    loading={loading} 
                    params={params}
                    onParamChange={handleParamChange}
                />
                <div className="status-indicator">
                    <span className={`status-dot ${loading ? 'pulsing' : 'idle'}`}></span>
                    <p>{status}</p>
                </div>
            </div>

            <div className="bento-card bento-metrics-card">
                <MetricsDashboard metrics={metrics} />
            </div>

            <div className="bento-card bento-explanation-card">
                <AlgorithmExplanation />
            </div>
        </div>

      </main>
    </div>
  );
}

export default App;

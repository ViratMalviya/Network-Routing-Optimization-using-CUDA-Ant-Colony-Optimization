import React, { useRef, useEffect } from 'react';

/**
 * MetricsDashboard — Displays execution metrics, convergence chart,
 * best path, and algorithm parameters after an optimization run.
 */
const MetricsDashboard = ({ metrics }) => {
    const chartRef = useRef(null);

    // Draw convergence chart on canvas
    useEffect(() => {
        if (!metrics || !metrics.convergence || metrics.convergence.length === 0) return;
        const canvas = chartRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';

        const w = rect.width;
        const h = rect.height;
        const padding = { top: 16, right: 16, bottom: 34, left: 52 };
        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        const data = metrics.convergence.filter(v => v !== null);
        if (data.length === 0) return;

        const maxVal = Math.max(...data) * 1.1;
        const minVal = Math.min(...data) * 0.9;
        const range = maxVal - minVal || 1;

        // Clear
        ctx.clearRect(0, 0, w, h);

        // Grid lines (horizontal only, subtle contrast)
        ctx.strokeStyle = '#e2e1dd';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i / 4) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();
        }

        // Draw Y and X axis borders (axes lines - high contrast)
        ctx.strokeStyle = '#555555';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        // Y Axis line
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + chartH);
        // X Axis line
        ctx.lineTo(w - padding.right, padding.top + chartH);
        ctx.stroke();

        // Draw ticks
        ctx.strokeStyle = '#555555';
        ctx.lineWidth = 1.0;
        // Y-axis ticks
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i / 4) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left - 4, y);
            ctx.lineTo(padding.left, y);
            ctx.stroke();
        }
        // X-axis ticks
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * chartW;
            ctx.beginPath();
            ctx.moveTo(x, padding.top + chartH);
            ctx.lineTo(x, padding.top + chartH + 4);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = '#444444';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 4; i++) {
            const val = maxVal - (i / 4) * range;
            const y = padding.top + (i / 4) * chartH;
            ctx.fillText(val.toFixed(1), padding.left - 6, y);
        }

        // Y-axis title (rotated vertically)
        ctx.save();
        ctx.translate(14, padding.top + chartH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#141414';
        ctx.font = '600 9px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('PATH COST', 0, 0);
        ctx.restore();

        // X-axis labels
        ctx.fillStyle = '#444444';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * chartW;
            const label = ((i + 1) * 10).toString();
            ctx.fillText(label, x, padding.top + chartH + 6);
        }

        // X-axis title
        ctx.fillStyle = '#141414';
        ctx.font = '600 9px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText('ITERATION', padding.left + chartW / 2, h - 4);

        // Gradient fill under curve
        const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
        gradient.addColorStop(0, 'rgba(255, 77, 0, 0.15)');
        gradient.addColorStop(1, 'rgba(255, 77, 0, 0.01)');

        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top + chartH);
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * chartW;
            const y = padding.top + ((maxVal - data[i]) / range) * chartH;
            ctx.lineTo(x, y);
        }
        ctx.lineTo(padding.left + chartW, padding.top + chartH);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Line
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * chartW;
            const y = padding.top + ((maxVal - data[i]) / range) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#ff4d00';
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.stroke();

        // Data points
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * chartW;
            const y = padding.top + ((maxVal - data[i]) / range) * chartH;

            // Outer glow
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 77, 0, 0.15)';
            ctx.fill();

            // Dot
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#ff4d00';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

    }, [metrics]);

    if (!metrics) {
        return (
            <div className="metrics-dashboard">
                <h2>Optimization Metrics</h2>
                <p className="metrics-empty">Run an optimization to see results.</p>
            </div>
        );
    }

    return (
        <div className="metrics-dashboard">
            <h2>Optimization Metrics</h2>

            {/* Stat Cards Row */}
            <div className="metrics-stat-row">
                <div className="metrics-stat-card">
                    <span className="metrics-stat-icon">⏱</span>
                    <div className="metrics-stat-data">
                        <span className="metrics-stat-value">
                            {metrics.execution_time_ms != null ? `${metrics.execution_time_ms}` : '—'}
                        </span>
                        <span className="metrics-stat-label">Time (ms)</span>
                    </div>
                </div>
                <div className="metrics-stat-card">
                    <span className="metrics-stat-icon">🎯</span>
                    <div className="metrics-stat-data">
                        <span className="metrics-stat-value">
                            {metrics.best_cost != null ? metrics.best_cost : '—'}
                        </span>
                        <span className="metrics-stat-label">Best Cost</span>
                    </div>
                </div>
                <div className="metrics-stat-card">
                    <span className="metrics-stat-icon">🔁</span>
                    <div className="metrics-stat-data">
                        <span className="metrics-stat-value">
                            {metrics.iterations_run || '—'}
                        </span>
                        <span className="metrics-stat-label">Iterations</span>
                    </div>
                </div>
            </div>

            {/* Best Path */}
            {metrics.best_path && metrics.best_path.length > 0 && (
                <div className="metrics-best-path">
                    <span className="metrics-section-label">Optimal Path</span>
                    <div className="metrics-path-nodes">
                        {metrics.best_path.map((n, i) => (
                            <React.Fragment key={i}>
                                <span className="metrics-path-node">N{n}</span>
                                {i < metrics.best_path.length - 1 && (
                                    <span className="metrics-path-arrow">→</span>
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            )}

            {/* Convergence Chart */}
            {metrics.convergence && metrics.convergence.length > 0 && (
                <div className="metrics-convergence">
                    <span className="metrics-section-label">Convergence</span>
                    <div className="metrics-chart-container">
                        <canvas ref={chartRef}></canvas>
                    </div>
                    <div className="chart-legend">
                        <span className="legend-item"><span className="legend-dot x-axis"></span><strong>X-Axis:</strong> Iterations</span>
                        <span className="legend-item"><span className="legend-dot y-axis"></span><strong>Y-Axis:</strong> Path Cost (Lower = Better)</span>
                    </div>
                    <p className="chart-description">
                        This graph shows how the global best path cost decreases and converges over iterations. As ants explore the network and lay down pheromones, the shortest path becomes reinforced.
                    </p>
                </div>
            )}

            {/* Parameters Table */}
            {metrics.parameters && (
                <div className="metrics-params">
                    <span className="metrics-section-label">Parameters (Hover for Details)</span>
                    <div className="metrics-params-grid">
                        <div className="metrics-param tooltip-container">
                            <span className="metrics-param-key">α</span>
                            <span className="metrics-param-val">{metrics.parameters.alpha}</span>
                            <span className="tooltip-text">
                                <strong>α (Alpha - Pheromone Weight):</strong> Controls how much ants are influenced by pheromone trails left by previous ants. Higher values lead to stronger path reinforcement.
                            </span>
                        </div>
                        <div className="metrics-param tooltip-container">
                            <span className="metrics-param-key">β</span>
                            <span className="metrics-param-val">{metrics.parameters.beta}</span>
                            <span className="tooltip-text">
                                <strong>β (Beta - Distance Weight):</strong> Controls how much ants prioritize shorter physical paths (heuristic information). Higher values make ants prefer physically shorter edges.
                            </span>
                        </div>
                        <div className="metrics-param tooltip-container">
                            <span className="metrics-param-key">ρ</span>
                            <span className="metrics-param-val">{metrics.parameters.rho}</span>
                            <span className="tooltip-text">
                                <strong>ρ (Rho - Evaporation Rate):</strong> The rate at which pheromones evaporate in each iteration. High evaporation (e.g. 0.9) helps forget bad routes quickly but may slow learning.
                            </span>
                        </div>
                        <div className="metrics-param tooltip-container">
                            <span className="metrics-param-key">Q</span>
                            <span className="metrics-param-val">{metrics.parameters.Q}</span>
                            <span className="tooltip-text">
                                <strong>Q (Pheromone Constant):</strong> The scaling constant for pheromone deposition. An ant deposits Q / L pheromones on its path (where L is the path cost). Higher Q strengthens successful paths faster.
                            </span>
                        </div>
                        <div className="metrics-param tooltip-container">
                            <span className="metrics-param-key">Ants</span>
                            <span className="metrics-param-val">{metrics.parameters.num_ants}</span>
                            <span className="tooltip-text">
                                <strong>Ants (Swarm Size):</strong> The number of simulated ants running through the network per iteration. More ants increase exploration but require more compute.
                            </span>
                        </div>
                        <div className="metrics-param tooltip-container">
                            <span className="metrics-param-key">Iter</span>
                            <span className="metrics-param-val">{metrics.parameters.iterations}</span>
                            <span className="tooltip-text">
                                <strong>Iterations:</strong> The total number of iterations the algorithm will run to reach convergence.
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MetricsDashboard;

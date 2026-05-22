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
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';

        const w = rect.width;
        const h = rect.height;
        const padding = { top: 20, right: 16, bottom: 28, left: 40 };
        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        const data = metrics.convergence.filter(v => v !== null);
        if (data.length === 0) return;

        const maxVal = Math.max(...data) * 1.1;
        const minVal = Math.min(...data) * 0.9;
        const range = maxVal - minVal || 1;

        // Clear
        ctx.clearRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = '#e8e8e6';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i / 4) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = '#999';
        ctx.font = '500 9px "Inter", sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 4; i++) {
            const val = maxVal - (i / 4) * range;
            const y = padding.top + (i / 4) * chartH;
            ctx.fillText(val.toFixed(1), padding.left - 6, y);
        }

        // X-axis labels
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1 || 1)) * chartW;
            const label = ((i + 1) * 10).toString();
            ctx.fillText(label, x, h - padding.bottom + 8);
        }

        // X-axis title
        ctx.fillStyle = '#aaa';
        ctx.font = '500 8px "Inter", sans-serif';
        ctx.fillText('Iteration', w / 2, h - 4);

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
                </div>
            )}

            {/* Parameters Table */}
            {metrics.parameters && (
                <div className="metrics-params">
                    <span className="metrics-section-label">Parameters</span>
                    <div className="metrics-params-grid">
                        <div className="metrics-param">
                            <span className="metrics-param-key">α</span>
                            <span className="metrics-param-val">{metrics.parameters.alpha}</span>
                        </div>
                        <div className="metrics-param">
                            <span className="metrics-param-key">β</span>
                            <span className="metrics-param-val">{metrics.parameters.beta}</span>
                        </div>
                        <div className="metrics-param">
                            <span className="metrics-param-key">ρ</span>
                            <span className="metrics-param-val">{metrics.parameters.rho}</span>
                        </div>
                        <div className="metrics-param">
                            <span className="metrics-param-key">Q</span>
                            <span className="metrics-param-val">{metrics.parameters.Q}</span>
                        </div>
                        <div className="metrics-param">
                            <span className="metrics-param-key">Ants</span>
                            <span className="metrics-param-val">{metrics.parameters.num_ants}</span>
                        </div>
                        <div className="metrics-param">
                            <span className="metrics-param-key">Iter</span>
                            <span className="metrics-param-val">{metrics.parameters.iterations}</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MetricsDashboard;

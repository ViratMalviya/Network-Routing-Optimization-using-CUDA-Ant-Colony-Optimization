import React from 'react';

const Controls = ({ onOptimize, loading, params, onParamChange }) => {
    return (
        <div className="controls-panel">
            <h2>Optimization Controls</h2>
            <p className="description">
                Configure ACO parameters and run the optimization. 
                The system will simulate ant paths across the network, updating pheromones over time.
            </p>

            {/* Parameter Sliders */}
            <div className="param-sliders">
                <div className="param-slider-group">
                    <div className="param-slider-header">
                        <label>α <span className="param-hint">Pheromone Influence</span></label>
                        <span className="param-value">{params.alpha}</span>
                    </div>
                    <input 
                        type="range" 
                        min="0.5" max="3.0" step="0.5"
                        value={params.alpha}
                        onChange={e => onParamChange('alpha', parseFloat(e.target.value))}
                        disabled={loading}
                        className="slider"
                    />
                </div>

                <div className="param-slider-group">
                    <div className="param-slider-header">
                        <label>β <span className="param-hint">Distance Influence</span></label>
                        <span className="param-value">{params.beta}</span>
                    </div>
                    <input 
                        type="range" 
                        min="0.5" max="5.0" step="0.5"
                        value={params.beta}
                        onChange={e => onParamChange('beta', parseFloat(e.target.value))}
                        disabled={loading}
                        className="slider"
                    />
                </div>

                <div className="param-slider-group">
                    <div className="param-slider-header">
                        <label>ρ <span className="param-hint">Evaporation Rate</span></label>
                        <span className="param-value">{params.rho}</span>
                    </div>
                    <input 
                        type="range" 
                        min="0.1" max="0.9" step="0.1"
                        value={params.rho}
                        onChange={e => onParamChange('rho', parseFloat(e.target.value))}
                        disabled={loading}
                        className="slider"
                    />
                </div>
            </div>

            <div className="button-group">
                <button 
                    onClick={onOptimize} 
                    disabled={loading}
                    className="btn btn-primary"
                >
                    {loading ? 'Running Optimization...' : 'Run CUDA Optimization'}
                </button>
            </div>
        </div>
    );
};

export default Controls;

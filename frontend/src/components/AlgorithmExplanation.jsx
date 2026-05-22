import React from 'react';

const AlgorithmExplanation = () => {
    return (
        <div className="explanation-content">
            <h2>Algorithm Breakdown</h2>
            <p>The <strong>Ant Colony Optimization (ACO)</strong> mimics how ants find the shortest path to food.</p>
            
            <div className="math-box">
                <h3>1. Path Selection Probability</h3>
                <div className="formula">
                    P<sub style={{fontSize: '0.7em'}}>ij</sub> = (τ<sub style={{fontSize: '0.7em'}}>ij</sub><sup>α</sup> × η<sub style={{fontSize: '0.7em'}}>ij</sub><sup>β</sup>) / Σ (τ<sup>α</sup> × η<sup>β</sup>)
                </div>
                <ul className="variables">
                    <li><strong>τ (Tau)</strong>: Pheromone amount on Edge(i, j). <em><br/>Higher = thick blue lines.</em></li>
                    <li><strong>η (Eta)</strong>: Heuristic (1 / Distance Cost).</li>
                    <li><strong>α, β</strong>: Weights prioritizing pheromone vs distance.</li>
                </ul>
            </div>

            <div className="math-box">
                <h3>2. Pheromone Update</h3>
                <div className="formula">
                    τ<sub style={{fontSize: '0.7em'}}>ij</sub> = (1 - ρ)τ<sub style={{fontSize: '0.7em'}}>ij</sub> + Δτ
                </div>
                <ul className="variables">
                    <li><strong>ρ (Rho)</strong>: Evaporation rate. <em><br/>Fades weak paths gradually.</em></li>
                    <li><strong>Δτ (Delta)</strong>: Deposited pheromone, calculated as <strong>Q / L</strong> (where L is the path cost).</li>
                    <li><strong>Q (Constant)</strong>: Pheromone deposit scaling constant. Higher Q increases the amount of pheromone deposited by successful ants, reinforcing shorter paths more aggressively.</li>
                </ul>
            </div>
            
            <p className="summary-text">
                <strong>Result:</strong> Over multiple iterations, evaporation destroys sub-optimal paths, while the shortest paths are compounded exponentially.
            </p>
        </div>
    );
};

export default AlgorithmExplanation;

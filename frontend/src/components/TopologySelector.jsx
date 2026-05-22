import React from 'react';

export const TOPOLOGIES = {
    default: {
        name: 'Default',
        description: '5-node directed graph',
        icon: '◇',
        nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }],
        edges: [
            { from: 0, to: 1, weight: 10 },
            { from: 0, to: 2, weight: 5 },
            { from: 1, to: 3, weight: 10 },
            { from: 2, to: 3, weight: 20 },
            { from: 2, to: 1, weight: 2 },
            { from: 3, to: 4, weight: 5 },
        ],
        startNode: 0,
        destNode: 4,
    },
    star: {
        name: 'Star',
        description: 'Hub-spoke network',
        icon: '✦',
        nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }],
        edges: [
            { from: 0, to: 1, weight: 4 },
            { from: 0, to: 2, weight: 7 },
            { from: 0, to: 3, weight: 3 },
            { from: 0, to: 4, weight: 6 },
            { from: 0, to: 5, weight: 5 },
            { from: 1, to: 0, weight: 4 },
            { from: 2, to: 0, weight: 7 },
            { from: 3, to: 0, weight: 3 },
            { from: 4, to: 0, weight: 6 },
            { from: 5, to: 0, weight: 5 },
            // Cross-links for interesting ACO paths
            { from: 1, to: 2, weight: 9 },
            { from: 2, to: 4, weight: 10 },
            { from: 3, to: 5, weight: 4 },
            { from: 5, to: 4, weight: 3 },
        ],
        startNode: 1,
        destNode: 4,
    },
    ring: {
        name: 'Ring',
        description: 'Circular daisy-chain',
        icon: '○',
        nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }],
        edges: [
            // Clockwise
            { from: 0, to: 1, weight: 3 },
            { from: 1, to: 2, weight: 5 },
            { from: 2, to: 3, weight: 4 },
            { from: 3, to: 4, weight: 6 },
            { from: 4, to: 5, weight: 2 },
            { from: 5, to: 0, weight: 7 },
            // Counter-clockwise
            { from: 1, to: 0, weight: 3 },
            { from: 2, to: 1, weight: 5 },
            { from: 3, to: 2, weight: 4 },
            { from: 4, to: 3, weight: 6 },
            { from: 5, to: 4, weight: 2 },
            { from: 0, to: 5, weight: 7 },
        ],
        startNode: 0,
        destNode: 3,
    },
    mesh: {
        name: 'Mesh',
        description: 'Fully connected',
        icon: '⬡',
        nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }],
        edges: [
            { from: 0, to: 1, weight: 8 },
            { from: 0, to: 2, weight: 5 },
            { from: 0, to: 3, weight: 12 },
            { from: 0, to: 4, weight: 15 },
            { from: 1, to: 2, weight: 3 },
            { from: 1, to: 3, weight: 7 },
            { from: 1, to: 4, weight: 10 },
            { from: 2, to: 3, weight: 6 },
            { from: 2, to: 4, weight: 9 },
            { from: 3, to: 4, weight: 4 },
        ],
        startNode: 0,
        destNode: 4,
    },
    tree: {
        name: 'Tree',
        description: 'Hierarchical layers',
        icon: '△',
        nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }, { id: 6 }],
        edges: [
            { from: 0, to: 1, weight: 5 },
            { from: 0, to: 2, weight: 8 },
            { from: 1, to: 3, weight: 3 },
            { from: 1, to: 4, weight: 6 },
            { from: 2, to: 5, weight: 4 },
            { from: 2, to: 6, weight: 7 },
            // Cross-tree shortcuts
            { from: 3, to: 5, weight: 9 },
            { from: 4, to: 6, weight: 2 },
        ],
        startNode: 0,
        destNode: 6,
    },
    linear: {
        name: 'Linear',
        description: 'Bus backbone',
        icon: '━',
        layout: 'horizontal',
        nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }],
        edges: [
            // Sequential bus — each node connects only to its neighbor
            { from: 0, to: 1, weight: 4 },
            { from: 1, to: 2, weight: 6 },
            { from: 2, to: 3, weight: 3 },
            { from: 3, to: 4, weight: 5 },
            // Reverse
            { from: 1, to: 0, weight: 4 },
            { from: 2, to: 1, weight: 6 },
            { from: 3, to: 2, weight: 3 },
            { from: 4, to: 3, weight: 5 },
        ],
        startNode: 0,
        destNode: 4,
    },
};

const TopologySelector = ({ current, onSelect, disabled }) => {
    return (
        <div className="topology-selector">
            <span className="topology-label">Topology</span>
            <div className="topology-chips">
                {Object.entries(TOPOLOGIES).map(([key, topo]) => (
                    <button
                        key={key}
                        className={`topology-chip ${current === key ? 'active' : ''}`}
                        onClick={() => onSelect(key, topo)}
                        disabled={disabled}
                        title={topo.description}
                    >
                        <span className="topology-chip-icon">{topo.icon}</span>
                        <span className="topology-chip-name">{topo.name}</span>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default TopologySelector;

# -*- coding: utf-8 -*-
"""
Created on Thu May 12 00:11:10 2022

@author: Fidel
"""

# Standard libraries
import networkx as nx


class EntangleBase:
    """Parent class for device entangled state analysis"""
    
    def __init__(self, backend):
        self.backend = backend
        
        self.device_name = backend.properties().backend_name
        self.nqubits = len(backend.properties().qubits)
        self.graph, self.connections, self.edge_params = self.gen_graph()
        self.edge_list = sorted(list(self.edge_params.keys()), key=lambda q: (q[0], q[1]))
        self.nedges = len(self.edge_params)
    
    def gen_graph(self):
        """
        """
        graph = nx.Graph()
        connections = {}
        edges = {}
        for i in range(self.nqubits):
            connections[i] = []
        # Iterate over possible cnot connections
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
                q0 = gate.qubits[0]
                q1 = gate.qubits[1]
                connections[q0].append(q1)
                if q0 < q1:
                    graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                    edges[q0, q1] = gate.parameters[0].value
        # Sort adjacent qubit list in ascending order
        for q in connections:
            connections[q].sort()
            
        return graph, connections, edges
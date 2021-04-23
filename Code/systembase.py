# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:03:25 2021

@author: Fidel
"""

# Standard libraries
import numpy as np

class SystemBase:
    """Parent class for device entangled state analysis"""
    def __init__(self, backend):
        self.backend = backend
        self.device_name = backend.properties().backend_name
        self.nqubits = len(backend.properties().qubits)
        self.edges = self.get_edges()
        self.nedges = len(self.edges)
        self.qubit_connections = self.get_qubit_connections()
        
    def get_edges(self):
        """
        Get a sorted list of unique physical edges, where each edge is a two 
        element tuple corresponding to qubit pairs
        """
        edges = []
        edges_matrix = np.array([], dtype=np.int32).reshape(0,2)
        # Iterate over possible cnot connections to construct an array of edges
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
                if gate.qubits[0] < gate.qubits[1]:
                    edges_matrix = np.vstack([edges_matrix, gate.qubits])
        # Sort 2d edges matrix by first qubit value then second qubit value
        edges_matrix = edges_matrix[np.lexsort((edges_matrix[:,1], edges_matrix[:,0]))]
        # Convert 2d numpy array to Python list of tuples
        for i in range(len(edges_matrix)):
            edges.append((edges_matrix[i,0], edges_matrix[i,1]))
        return edges
    
    def get_qubit_connections(self):
        """
        Get a dictionary of qubit connections, where keys are qubits and
        values are adjacent qubits
        """
        qubit_connections = {}
        for i in range(self.nqubits):
            adjacent = []
            for edge in self.edges:
                if i == edge[0]:
                    adjacent.append(edge[1])
                elif i == edge[1]:
                    adjacent.append(edge[0])
            qubit_connections[i] = adjacent
        return qubit_connections
    
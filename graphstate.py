# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:12:46 2021

@author: Fidel
"""

# Standard libraries
import numpy as np

# Qiskit libraries
from qiskit import QuantumCircuit

class GraphState:
    """
    """
    def __init__(self, backend):
        self.backend = backend
        self.nqubits = len(backend.properties().qubits)
        self.edges = self.get_edges()
        self.nedges = len(self.edges)
        self.tomography_targets = self.get_tomography_targets()
        self.graphstate_circuit = self.gen_graphstate()
        
    def get_edges(self):
        """
        Get a sorted list of physical edges, where each edge is a two element 
        tuple corresponding to qubit pairs
        """
        edges = []
        edges_matrix = np.array([], dtype=np.int16).reshape(0,2)
        # Iterate over possible cnot connections to construct a matrix of edges
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
                if gate.qubits[0] < gate.qubits[1]:
                    edges_matrix = np.vstack([edges_matrix, gate.qubits])
        edges_matrix = edges_matrix[np.lexsort((edges_matrix[:,1], edges_matrix[:,0]))]
        # Convert 2d numpy array to Python list of tuples
        for i in range(len(edges_matrix)):
            edges.append((edges_matrix[i,0], edges_matrix[i,1]))
        return edges
    
    def get_tomography_targets(self):
        """
        Get a dictionary of tomography targets with target edges as keys and
        and neighbouring edges as values
        """
        tomography_targets = {}
        for edge in self.edges:
            other_edges = self.edges.copy()
            other_edges.remove(edge)
            connected_edges = []
            for edgej in other_edges:
                if (edge[0] in edgej) | (edge[1] in edgej):
                    connected_edges.append(edgej)
            tomography_targets[edge] = connected_edges
        return tomography_targets

    def gen_graphstate(self):
        """
        Generate a graph state circuit for the whole IBM device
        """
        graphstate = QuantumCircuit(self.nqubits)
        unconnected_edges = self.edges.copy()
        # Apply Hadamard gates to every qubit
        graphstate.h(list(range(self.nqubits)))
        # Connect every edge with cz gates
        while unconnected_edges:
            connected_qubits = [] # Qubits connected in the same time step
            remove = []
            for edge in unconnected_edges:
                if (edge[0] in connected_qubits) | (edge[1] in connected_qubits) == False:
                    graphstate.cz(edge[0], edge[1])
                    connected_qubits.append(edge[0])
                    connected_qubits.append(edge[1])
                    remove.append(edge)
            # Remove connected edges
            for edge in remove:
                unconnected_edges.remove(edge)   
        return graphstate
    
    def gen_tomography_circuits(self):
        pass
    


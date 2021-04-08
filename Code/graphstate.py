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
    Construct a graph state circuit over every physical edge in the device 
    and perform quantum state tomography over each edge qubit pair
    """
    def __init__(self, backend):
        # Initialise of input backend
        self.backend = backend
        self.device_name = backend.properties().backend_name
        self.nqubits = len(backend.properties().qubits)
        self.edges = self.get_edges()
        self.nedges = len(self.edges)
        self.tomography_targets = self.get_tomography_targets()
        self.tomography_batches = self.get_tomography_batches()
        self.nbatches = len(self.tomography_batches)
        self.graphstate_circuit = self.gen_graphstate_circuit()
        
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
    
    def get_tomography_targets(self):
        """
        Get a dictionary of tomography targets, where keys are target edges and
        values are neighbouring edges
        """
        tomography_targets = {}
        # Iterate over every edge
        for edge in self.edges:
            other_edges = self.edges.copy()
            other_edges.remove(edge)
            connected_edges = []
            # Iterate over all other edges
            for edgej in other_edges:
                if np.any(np.isin(edge, edgej)):
                    connected_edges.append(edgej)
            tomography_targets[edge] = connected_edges
        return tomography_targets

    def get_tomography_batches(self):
        """
        Get a dictionary of tomography batches, where keys are batch numbers
        and values are lists of target edges
        """
        batches = {}
        unbatched_edges = self.edges.copy()
        i = 0
        while unbatched_edges:
            batches[f'batch{i}'] = []
            batched_qubits = []
            remove = []
            for edge in unbatched_edges:
                qubits = sum(self.tomography_targets[edge], ())
                # Append edge to batch only if target and adjacent qubits have 
                # not been batched in the current cycle
                if np.any(np.isin(qubits, batched_qubits)) == False:
                    batches[f'batch{i}'].append(edge)
                    batched_qubits.extend(sum(self.tomography_targets[edge], ()))
                    remove.append(edge)
            for edge in remove:
                unbatched_edges.remove(edge)
            i += 1
        return batches

    def gen_graphstate_circuit(self):
        """
        Generate a graph state circuit over every physical edge
        """
        circuit = QuantumCircuit(self.nqubits)
        unconnected_edges = self.edges.copy()
        # Apply Hadamard gates to every qubit
        circuit.h(list(range(self.nqubits)))
        # Connect every edge with cz gates
        while unconnected_edges:
            connected_qubits = [] # Qubits already connected in the current time step
            remove = []
            for edge in unconnected_edges:
                if np.any(np.isin(edge, connected_qubits)) == False:
                    circuit.cz(edge[0], edge[1])
                    connected_qubits.extend(edge)
                    remove.append(edge)
            # Remove connected edges from unconnected edges list
            for edge in remove:
                unconnected_edges.remove(edge)   
        return circuit
    
    def gen_tomography_circuits(self):
        pass


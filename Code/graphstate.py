# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:12:46 2021

@author: Fidel
"""

# Standard libraries
import numpy as np

# Qiskit libraries
from qiskit import QuantumCircuit

# Local libraries
from systembase import SystemBase

class GraphState(SystemBase):
    """
    Construct a graph state circuit over every physical edge in the device 
    and perform quantum state tomography over each edge qubit pair
    """
    def __init__(self, backend):
        super().__init__(backend)
        self.tomography_targets = self.get_tomography_targets()
        self.tomography_batches = self.get_tomography_batches()
        self.nbatches = len(self.tomography_batches)
        self.graphstate_circuit = self.gen_graphstate_circuit()
    
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


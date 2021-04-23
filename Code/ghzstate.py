# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:33:02 2021

@author: Fidel
"""

# Standard libraries
import numpy as np

# Qiskit libraries
from qiskit import QuantumCircuit, IBMQ

# Local libraries
from systembase import SystemBase
from utilities import startup, check_provider

class GHZState(SystemBase):
    """
    Construct a
    """
    def __init__(self, backend):
        # Initialise from input backend
        super().__init__(backend)
    
    def get_paths(self, n):
        """
        Get a list of all possible qubit paths of length n
        """
        paths = []
        for qubit in self.qubit_connections:
            for connection in self.qubit_connections[qubit]:
                paths.append([qubit, connection])
        for i in range(n-3):
            new_paths = []
            for path in paths:
                for connection in self.qubit_connections[path[-1]]:
                    if connection not in path:
                        new_path = path.copy()
                        new_path.append(connection)
                        new_paths.append(new_path)
            paths = new_paths
        final_paths = []
        for path in paths:
            for connection in self.qubit_connections[path[-1]]:
                if (connection not in path) & (connection > path[0]):
                    new_path = path.copy()
                    new_path.append(connection)
                    final_paths.append(tuple(new_path))
        return final_paths
                

if __name__ == '__main__':
    startup()
    provider = IBMQ.get_provider('ibm-q-melbourne')
    backend = provider.get_backend('ibmq_manhattan')
    test = GHZState(backend)
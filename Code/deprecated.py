# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:34:56 2021

@author: Fidel
"""

import numpy as np

def get_edges(backend):
    """
    Get an nx2 array of edges where columns are qubit numbers (vertices), and
    then sorts them by 1st column values, then 2nd column values
    """
    edges = np.array([], dtype=np.int32).reshape(0,2)
    for gate in backend.properties().gates:
        if gate.gate == 'cx':
            if gate.qubits[0] < gate.qubits[1]:
                edges = np.vstack([edges, gate.qubits])
    edges = edges[np.lexsort((edges[:,1], edges[:,0]))]
    return edges

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 00:41:16 2021

@author: Fidel
"""
# Standard libraries
import numpy as np
import pandas as pd

# Qiskit libraries

# Local modules
from graphstate import GraphState

def gen_devices_df(provider):
    columns = ['Device name', 'Number of qubits', 'Number of edges', 'Number of batches', \
               'Edges', 'Tomography targets', 'Tomography batches', 'Graph state circuit']
    data = []
    for backend in provider.backends():
        properties = backend.properties()
        try:
            graphstate = GraphState(backend)
            entry = [graphstate.device_name, graphstate.nqubits, graphstate.nedges, \
                     graphstate.nbatches, graphstate.edges, graphstate.tomography_targets, \
                     graphstate.tomography_batches, graphstate.graphstate_circuit]
            data.append(entry)
        except:
            pass
    devices_df = pd.DataFrame(data, columns=columns)
    devices_df = devices_df.sort_values('Number of qubits', ascending=False).reset_index(drop=True)
    return devices_df
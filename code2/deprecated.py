# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 00:41:16 2021

@author: Fidel
"""
# Standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Qiskit libraries

# Local modules
from entangledstates import GraphState

def gen_devices_df(provider):
    columns = ['Device name', 'Number of qubits', 'Number of edges', 'Number of batches', \
               'Graph', 'Edges','Tomography targets', 'Tomography batches', 'Graph state circuit']
    data = []
    for backend in provider.backends():
        try:
            graphstate = GraphState(backend)
            entry = [graphstate.device_name, graphstate.nqubits, graphstate.nedges, \
                     graphstate.nbatches, graphstate.connections, graphstate.edge_list, \
                     graphstate.tomography_targets, graphstate.tomography_batches, \
                     graphstate.graphstate_circuit]
            data.append(entry)
        except:
            print('pass')
    devices_df = pd.DataFrame(data, columns=columns)
    devices_df = devices_df.sort_values('Number of qubits', ascending=False).reset_index(drop=True)
    return devices_df

def plot_negativities(negativities, name=None, size=(8, 6)):
    """"""
    
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in negativities.keys()])
    Y = np.array(list(negativities.values()))
    idx = Y.argsort()
    X = X[idx]
    Y = Y[idx]
        
    fig, ax = plt.subplots(figsize=size, dpi=144)
    ax.scatter(X, Y)
    
    #ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    
    ax.set_xlabel("Qubit Pairs")
    ax.set_ylabel("Negativity")
    ax.set_title(f"Graph-State Negativities ({name})")
    
    return X, Y

if __name__ == "__main__":
    #plot_negativities(negativities)
    pass
    
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:23:27 2022

@author: Fidel
"""

# Standard libraries
import matplotlib.pyplot as plt
import numpy as np

# Qiskit
from qiskit import IBMQ

# Local modules    
from graphstate import GraphState

    
def plot_negativities(GraphStateBackend, size=(6.4, 4.8)):
    """"""
    
    # Figure
    fig, ax = plt.subplots(figsize=size)
    
    # Get unmitigated negativities
    n_mean, n_all = GraphStateBackend.get_negativities(mit=False)
    # Qubit pair edges on x-axis, negativities on Y-axis with std error
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in n_mean.keys()])
    Y0 = np.array(list(n_mean.values()))
    Y0err = np.array([np.std(list(bns.values())) for bns in n_all.values()])
    Y0min = Y0 - Y0err
    
    # Get mitigated results if they exist
    if GraphStateBackend.qrem is True:
        n_mean_mit, n_all_mit = GraphStateBackend.get_negativities(mit=True)
        Y1 = np.array(list(n_mean_mit.values()))
        Y1err = np.array([np.std(list(bns.values())) for bns in n_all_mit.values()])
        Y1min = Y1 - Y1err
        # Order in ascending (lowest) negativitity
        idx = Y1min.argsort()
        X = X[idx]
        Y1 = Y1[idx]
        Y1err = Y1err[idx]
        # Error bars
        ax.errorbar(X, Y1, yerr=Y1err, capsize=3, fmt='.', c='b', label='QREM')
        
    else:
        idx = Y0min.argsort()
        X = X[idx]
    
    # Order based on unmitigated results
    Y0 = Y0[idx]
    Y0err = Y0err[idx]
    # Error bars
    ax.errorbar(X, Y0, yerr=Y0err, capsize=3, fmt='.', c='r', label='No QREM')
    
    # Fig params
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    ax.legend()
    
    name = GraphStateBackend.device_name
    ax.set_xlabel("Qubit Pairs")
    ax.set_ylabel("Negativity")
    ax.set_title(f"Native-graph state negativities ({name})")
    
    return fig

def plot_dev_batches(hub='ibm-q-melbourne', size=(6.4, 4.8)):
    """"""
    # Figure
    fig, ax = plt.subplots(figsize=size)
    
    provider = IBMQ.get_provider(hub)
    X = []
    Y = []
    N = []
    
    for backend in provider.backends():
        try:
            properties = backend.properties()
            nqubits = len(properties.qubits)
            name = properties.backend_name
            
            nbatches = len(GraphState(backend).gen_batches())
            
            X.append(name + f', {nqubits}')
            Y.append(nbatches)
            N.append(nqubits)
            
        except:
            pass
    # Convert to numpy arrays for sorting
    X = np.array(X)
    Y = np.array(Y)
    N = np.array(N)
    # Sort by number of qubits
    idx = N.argsort()
    X = X[idx]
    Y = Y[idx]
    
    # Plot
    ax.scatter(X, Y)
    ax.tick_params(axis='x', labelrotation=90)
        
    return fig
    
    
def hrange(values):
    return 0.5*(max(values) - min(values))
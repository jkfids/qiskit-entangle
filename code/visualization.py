# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:36:23 2023

@author: jfide
"""

# Standard
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import networkx as nx

# Local
from graphstate import GraphState, calc_n_mean, filter_edges
from qiskit.visualization import plot_gate_map, plot_coupling_map
from qubit_coords import qubit_coords127

# GRAPH STATE

def plot_negativities_multi(backend, n_list, nmit_list=None, figsize=(6.4, 4.8), idx=None, return_idx=False):
    """
    Plot average negativity across multiple experiments with error bars as std

    """

    # Figure
    fig, ax = plt.subplots(figsize=figsize)


    # Extract the mean negativity and its standard deviation
    while True:
        try:
            edges = n_list[0].keys()
            n_mean, n_std = calc_n_mean(n_list)
        except:
            n_list = [n_list]
            if nmit_list is not None:
                nmit_list = [nmit_list]
        else:
            break

    # Convert into array for plotting
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in edges])
    Y0 = np.fromiter(n_mean.values(), float)
    Y0err = np.fromiter(n_std.values(), float)

    # If mitigated results are included
    try:
        nmit_mean, nmit_std = calc_n_mean(nmit_list)

        Y1 = np.fromiter(nmit_mean.values(), float)
        Y1err = np.fromiter(nmit_std.values(), float)
        # Order in increasing minimum negativity (QREM)
        Y1min = Y1 - Y1err
        if idx is None:
            idx = Y1min.argsort()
        Y1 = Y1[idx]
        Y1err = Y1err[idx]
    except:
        # Order in increasing minimum negativity (No QREM)
        Y0min = Y0 - Y0err
        if idx is None:
            idx = Y0min.argsort()

    X = X[idx]
    Y0 = Y0[idx]
    Y0err = Y0err[idx]

    # Plot
    ax.errorbar(X, Y0, yerr=Y0err, capsize=3, fmt='.', c='r', 
                label=f'No QREM (Mean negativity: {np.mean(Y0):.4f})')
    try:
        ax.errorbar(X, Y1, yerr=Y1err, capsize=3, fmt='.', c='b', 
                    label=f'QREM (Mean negativity: {np.mean(Y1):.4f})')
    except:
        pass

    # Fig params
    ax.set_yticks(np.arange(0, 0.55, 0.05))
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    ax.legend()

    ax.set_xlabel("Qubit Pairs")
    ax.set_ylabel("Negativity")
    #ax.set_title(f"Native-graph state negativities ({backend.name()})")
    ax.set_title(backend.name())
    fig.set_tight_layout(True)

    if return_idx:
        return fig, idx
    else:
        return fig

def plot_negativities127(backend, n_list, nmit_list, figsize=(14, 9)):
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)
    
    # Extract the mean negativity and its standard deviation
    edges = n_list[0].keys()
    n_mean, n_std = calc_n_mean(n_list)
    nmit_mean, nmit_std = calc_n_mean(nmit_list)
    
    # Convert into array for plotting
    X = np.array([f'{edge[0]}-{edge[1]}' for edge in edges])
    Y0 = np.fromiter(n_mean.values(), float)
    Y0err = np.fromiter(n_std.values(), float)
    
    Y1 = np.fromiter(nmit_mean.values(), float)
    Y1err = np.fromiter(nmit_std.values(), float)
    
    # Order in increasing minimum negativity (QREM)
    Y1min = Y1 - Y1err
    idx = Y1min.argsort()
    Y1 = Y1[idx]
    Y1err = Y1err[idx]
    
    X = X[idx]
    Y0 = Y0[idx]
    Y0err = Y0err[idx]
    
    hp = int(len(X)/2)
    
    ax1.errorbar(X[:hp], Y0[:hp], yerr=Y0err[:hp], capsize=3, fmt='.', c='r', 
                label=f'No QREM (Mean negativity: {np.mean(Y0):.4f})')
    
    ax1.errorbar(X[:hp], Y1[:hp], yerr=Y1err[:hp], capsize=3, fmt='.', c='b', 
                label=f'QREM (Mean negativity: {np.mean(Y1):.4f})')
    
    ax2.errorbar(X[hp:], Y0[hp:], yerr=Y0err[hp:], capsize=3, fmt='.', c='r', 
                label=f'No QREM (Mean negativity: {np.mean(Y0):.4f})')
    
    ax2.errorbar(X[hp:], Y1[hp:], yerr=Y1err[hp:], capsize=3, fmt='.', c='b', 
                label=f'QREM (Mean negativity: {np.mean(Y1):.4f})')
    
    for ax in (ax1, ax2):
        ax.set_yticks(np.arange(0, 0.55, 0.05))
        ax.tick_params(axis='x', labelrotation=90)
        ax.grid()
        ax.set_ylabel("Negativity")
        ax.margins(0.025, 0.05)
        
    ax1.legend()
    ax2.set_xlabel("Qubit Pairs")
    ax1.set_title(backend.name())
    fig.set_tight_layout(True)
    
    return fig

def plot_nmap127(graphstate, n_list):
    
    nqubits = 127
    
    n_mean, n_std = calc_n_mean(n_list)
    
    cmap = mpl.cm.get_cmap('RdBu')
    cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName",['r', 'm', 'b'])
    
    edges = filter_edges(n_mean, threshold=0.025)
    G = nx.Graph()
    G.add_edges_from(edges)
    unconnected = list(set(range(127)) - list(nx.connected_components(G))[0])
    
    qubit_n = np.zeros(nqubits)
    qubit_color = []
    edge_list = []
    line_color = []
    
    for key, values in n_mean.items():
        edge_list.append(key)
        line_color.append(mpl.colors.to_hex(cmap(2*values), keep_alpha=True))
        qubit_n[key[0]] += values
        qubit_n[key[1]] += values
        
    for i, n in enumerate(qubit_n):
        x = 2*n/graphstate.graph.degree[i]
        if i in unconnected:
            #qubit_color.append('#D3D3D3')
            qubit_color.append('#C0C0C0')
        else:
            qubit_color.append(mpl.colors.to_hex(cmap(x), keep_alpha=True))
            
    fig = plot_coupling_map(nqubits, qubit_coords127, edge_list, line_color=line_color, qubit_color=qubit_color, \
                            line_width=6, figsize=(12,12))
    
    norm = mpl.colors.Normalize(vmin=0, vmax=0.5)

    ax = fig.get_axes()[0]
    cax = fig.add_axes([0.9, 0.2, 0.015, 0.605])
    
    im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Negativity')
    fig.savefig('output/ghznmap127.png', dpi=400)
    
    return fig
    


def plot_device_nbatches(provider, size=(6.4, 4.8)):
    """Plot the number of QST patches for each available device"""

    # Figure
    fig, ax = plt.subplots(figsize=size)

    X = []  # Name
    Y = []  # No. of batches
    N = []  # No. of qubits

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


def plot_cxerr_corr(properties, adj_edges, n_mean, inc_adj=True, figsize=(6.4, 4.8)):
    """Plot negativity vs. CNOT error"""

    # Figure
    #fig, ax = plt.subplots(figsize=figsize)

    edges = n_mean.keys()
    
    if inc_adj is True:
        X = []
        for edge in edges:
            targ_err = [properties.gate_error('cx', edge)]
            adj_errs = [properties.gate_error('cx', adj_edge) 
                       for adj_edge in adj_edges[edge]
                       if adj_edge in edges]
            err = np.mean(targ_err + adj_errs)
            X.append(err)
        X = np.array(X)
    else:
        X = np.fromiter((properties.gate_error('cx', edge)
                         for edge in edges), float)

    Y = np.fromiter((n_mean.values()), float)

    #ax.scatter(X, Y)
    #ax.set_xlabel("CNOT Error")
    #ax.set_ylabel("Negativity")

    return X, Y


# GHZ STATE
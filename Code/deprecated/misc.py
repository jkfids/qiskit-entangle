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

class GHZState(SystemBase):
    """
    """
    def __init__(self, backend):
        # Initialise from input backend
        super().__init__(backend)
        
    def calc_error(self, q1, q2):
        """Calculate the natural log of the cnot error between adjacent qubits"""
        if q1 < q2:
            return np.log(self.edge_errors[q1, q2])
        else:
            return np.log(self.edge_errors[q2, q1])

    def get_paths(self, n):
        """
        Get a dictionary where keys are all possible qubit paths of length n
        and values are total path error
        """
        paths = {}
        # Get initial paths of length 2
        for qubit in self.qubit_connections:
            for connection in self.qubit_connections[qubit]:
                paths[qubit, connection] = self.calc_error(qubit, connection)
        # Iterate over path length
        for i in range(n-2):
            new_paths = {}
            for path, lnerror in paths.items():
                for connection in self.qubit_connections[path[-1]]:
                    # Filter out cyclical paths
                    if connection not in path:
                        if i < n-3:
                            new_path = list(path).copy()
                            new_path.append(connection)
                            new_paths[tuple(new_path)] \
                                = lnerror + self.calc_error(path[-1], connection)
                        elif connection > path[0]:
                            new_path = list(path).copy()
                            new_path.append(connection)
                            new_paths[tuple(new_path)] \
                                = lnerror + self.calc_error(path[-1], connection)
            paths = new_paths.copy()
        # Sort dictionary by total error in ascending order
        paths = dict(sorted(paths.items(), key=lambda item: item[1]))
        return paths
    
    def get_edges(self):
        """
        Get a sorted list of unique physical edges, where each edge is a two 
        element tuple corresponding to qubit pairs, as well as a dictionary
        of corresponding cnot errors
        """
        edges = []
        edge_errors = {}
        edges_matrix = np.array([]).reshape(0,3)
        # Iterate over possible cnot connections
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
                # Iterate over parameters to obtain cnot error
                for param in gate.parameters:
                    if param.name == 'gate_error':
                        edge = [gate.qubits[0], gate.qubits[1], param.value]
                        #if gate.qubits[0] < gate.qubits[1]:
                        edges_matrix = np.vstack([edges_matrix, edge])
        # Sort 2d edges matrix by first qubit value then second qubit value
        edges_matrix = edges_matrix[np.lexsort((edges_matrix[:,1], edges_matrix[:,0]))]
        # Convert 2d numpy array to list (of tupples) and dictionary
        for row in edges_matrix:
            edge = tuple(row[:2].astype(int))
            edge_errors[edge] = row[2]
            if edge[0] < edge[1]:
                edges.append(edge)
        return edges, edge_errors

    def gen_pathtree(self, source, nodes):
        """
        """
        dist = [self.nqubits]*self.nqubits
        path = [[] for i in range(self.nqubits)]
        depth = [None]*self.nqubits
        max_depth = [None]*self.nqubits
        
        dist[source] = 0
        path[source] = [source]
        depth[source] = 0
        max_depth[source] = 0
        
        unvisited = dict(zip(list(range(self.nqubits)), dist))
        
        for i in range(nodes):
            u = min(unvisited, key=unvisited.get)
            del unvisited[u]
            
            for v in self.connections[u]:
                alt = dist[u] + self.edges[tuple(sorted((u, v)))] 
                if alt < dist[v]:
                    unvisited[v] = dist[v] = alt
                    path[v] = path[u] + [v]
                
        return dist, path, unvisited
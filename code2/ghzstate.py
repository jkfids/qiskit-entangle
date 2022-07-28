# -*- coding: utf-8 -*-
"""
Created on Thu May 12 00:04:24 2022

@author: Fidel
"""

# Standard libraries
import numpy as np

# Local modules
from entanglebase import EntangleBase
from utilities import startup

def sum_error(a, b):
    return a + b - a*b

class GHZState(EntangleBase):
    
    def __init__(self, backend):
        
        super().__init__(backend)
        
    def gen_pathtree(self, source, nodes):
        """Modified Dijkstra's algorithm"""
        
        length = {q: self.nqubits for q in range(self.nqubits)}
        path = {q: [] for q in range(self.nqubits)}
        #degree_visited = {q: 0 for q in range(self.nqubits)}
        degree_unvisited = dict(self.graph.degree)
        error = {q: 1 for q in range(self.nqubits)}
        #tlength = 0
        
        length[source] = 0
        path[source] = [source]
        error[source] = 0
        
        visited = []
        unvisited = length.copy()
        
        for i in range(nodes):
            # Find minimum length (CNOT depth) unconnected nodes
            lmin = min(unvisited.values())
            unvisited_lmin = {key: value for key, value in unvisited.items() if value == lmin}
            u_lmin = [key for key, value in unvisited.items() if value == lmin]
            # Pick potential nodes with largest degree
            degree_lmin = {key: degree_unvisited[key] for key in u_lmin}
            dmax = max(degree_lmin.values())
            u_lmin_dmax = [key for key, value in degree_lmin.items() if value == dmax]
            # Pick node with lowest propagated CNOT error
            u = min(u_lmin_dmax, key=error.get)
            visited.append(u)
            del unvisited[u]
            
            try:
                u_prev = path[u][-2]
                #degree_visited[u_prev] += 1
                degree_unvisited[u_prev] -= 1
                for v_prev in self.connections[u_prev]:
                    try:
                        unvisited[v_prev] += 1
                        length[v_prev] = unvisited[v_prev]
                    except: pass
                #tlength += self.edge_params[tuple(sorted((u, path[u][-2])))]
            except: pass
            
            for v in self.connections[u]:
                alt = length[u] + 1
                if alt < length[v]:
                    unvisited[v] = length[v] = alt
                    path[v] = path[u] + [v]
                    error[v] = sum_error(error[u], self.edge_params[tuple(sorted((u, v)))])
        
        cnot_instr = {tuple(path[q][-2:]): length[q] for q in visited[1:]}
              
        return cnot_instr
    
            
    
if __name__ == "__main__":
    provider = startup()
    backend = provider.get_backend('ibmq_montreal')
    
    test = GHZState(backend)
    pathtree = test.gen_pathtree(13, 7)
    
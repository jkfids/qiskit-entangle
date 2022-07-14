# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:03:25 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import networkx as nx
#from numba import jit, njit

# Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, Aer, execute

# Local modules
from utilities import startup, pauli_n, bit_str_list

# Utility functions and globals

def str2arr(bit_string):
    return np.fromiter(bit_string, int)    

def arr2str(bit_array):
    return np.array2string(bit_array, separator="")[1:-1]

def padded_array(array, fill=0):
    lens = np.array([len(item) for item in array])
    filt = lens[:,None] > np.arange(lens.max())
    padded = fill * np.ones(filt.shape, dtype=int)
    padded[filt] = np.concatenate(array)
    return padded

def calc_N(rho_pt):
    """Calculate the negativity of entanglement for a given density matrix"""
    w, v = la.eig(rho_pt)
    N = np.sum(w[w<0])
    return abs(N)

def calc_ptrans(rho):
    """Calculate the partial transpose a 4x4 array (A kron B) w.r.t B"""
    rho_pt = np.zeros(rho.shape, dtype=complex)
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            rho_pt[i:i+2, j:j+2] = rho[i:i+2, j:j+2].transpose()
            
    return rho_pt

def splice_str(string, idx):
    return ''.join([string[i] for i in idx])


basis_list = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
ext_basis_list = ['II', 'IX', 'IY', 'IZ',
                  'XI', 'XX', 'XY', 'XZ',
                  'YI', 'YX', 'YY', 'YZ',
                  'ZI', 'ZX', 'ZY', 'ZZ']
basis_dict = {basis:{bit:0 for bit in bit_str_list(2)} for basis in basis_list}


# Main code

class System:
    """Parent class for device entangled state analysis"""
    def __init__(self, backend):
        self.backend = backend
        
        self.device_name = backend.properties().backend_name
        self.nqubits = len(backend.properties().qubits)
        self.graph, self.connections, self.edge_params = self.gen_graph()
        self.edge_list = sorted(list(self.edge_params.keys()), key=lambda q: (q[0], q[1]))
        self.nedges = len(self.edge_params)
    
    def gen_graph(self):
        """
        """
        graph = nx.Graph()
        connections = {}
        edges = {}
        for i in range(self.nqubits):
            connections[i] = []
        # Iterate over possible cnot connections
        for gate in self.backend.properties().gates:
            if gate.gate == 'cx':
                q0 = gate.qubits[0]
                q1 = gate.qubits[1]
                connections[q0].append(q1)
                if q0 < q1:
                    graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                    edges[q0, q1] = gate.parameters[0].value
        # Sort adjacent qubit list in ascending order
        for q in connections:
            connections[q].sort()
            
        return graph, connections, edges
    
            
                
class GraphState(System):
    """
    Construct a graph state circuit over every physical edge in the device 
    and perform quantum state tomography over each edge qubit pair
    """
    def __init__(self, backend):
        super().__init__(backend)
        
        if backend is not None:
            self.adj_edges, self.adj_qubits = self.get_adjs()
            self.batches = self.get_batches()
            self.nbatches = len(self.batches)
            self.graphstate_circuit = self.gen_graphstate_circuit()
            
    def set_backend(self, backend):
        self.__init__(backend)
    
    def get_adjs(self):
        """
        Get a dictionary of tomography targets, where keys are target edges and
        values are neighbouring edges. Also get a dictionary where keys are
        target edges and values are adjacent qubits.
        """
        adj_edges = {}
        adj_qubits = {}
        # Iterate over every edge
        for edge in self.edge_list:
            other_edges = self.edge_list.copy()
            other_edges.remove(edge)
            connected_edges = []
            connected_qubits = []
            # Iterate over all other edges
            for edgej in other_edges:
                if np.any(np.isin(edge, edgej)):
                    connected_edges.append(edgej)
                    for q in edgej:
                        if q not in edge: connected_qubits.append(q)
            adj_edges[edge] = connected_edges
            adj_qubits[edge] = connected_qubits
        return adj_edges, adj_qubits

    def get_batches(self):
        """
        Get a dictionary of tomography batches, where keys are batch numbers
        and values are lists of tomography groups (targets + adj qubits)
        """
        batches = {}
        unbatched_edges = self.edge_list.copy()
        i = 0
        while unbatched_edges:
            batches[f'batch{i}'] = []
            batched_qubits = []
            remove = []
            for edge in unbatched_edges:
                qubits = sum(self.adj_edges[edge], ())
                # Append edge to batch only if target and adjacent qubits have 
                # not been batched in the current cycle
                if np.any(np.isin(qubits, batched_qubits)) == False:
                    group = tuple(list(edge) + self.adj_qubits[edge])
                    batches[f'batch{i}'].append(group)
                    batched_qubits.extend(sum(self.adj_edges[edge], ()))
                    remove.append(edge)
            for edge in remove:
                unbatched_edges.remove(edge)
            i += 1
        return batches
    
# =============================================================================
#     def gen_qrem_circuits(self):
#         """
#         """
#         qrem_circuits = {}
#         
#         for batchn, targets in self.batches.items():
#             batch_circuits = {}
#             
#             targ_arr = np.array(targets, dtype=int)
#             adj_lsts = [self.adj_qubits[targ] for targ in targets]
#             adj_arr = padded_array(adj_lsts, -1)
#             qubit_arr = np.concatenate((targ_arr, adj_arr), 1)
#             
#             for bit_str in bit_str_list(qubit_arr.shape[1]):
#                 bit_arr = str2arr(bit_str)
#                 col = np.nonzero(bit_arr)[0]
#                 
#                 qubit_list = qubit_arr.flatten()
#                 x_list = qubit_arr[:, col].flatten()
#                 x_list = x_list[x_list != -1]
#                 
#                 circ = QuantumCircuit(self.nqubits, self.nqubits)
#                 circ.x(x_list)
#                 circ.measure()
#         
#         return qrem_circuits
# =============================================================================

    def run_qrem(self):
        """"""
        [circ0, circ1] = self.gen_qrem_circuits()
        
        

    def gen_qrem_circuits(self):
        """"""     

        circ0 = QuantumCircuit(self.nqubits, self.nqubits)
        circ0.measure_all()
        
        circ1 = QuantumCircuit(self.nqubits, self.nqubits)
        circ1.x(range(self.nqubits))
        circ1.measure_all()
        
        return [circ0, circ1]
    
    def run_qrem_circuits(self):
        """"""
        pass
    

    def gen_graphstate_circuit(self):
        """
        Generate a graph state circuit over every physical edge
        """
        circuit = QuantumCircuit(self.nqubits, self.nqubits)
        unconnected_edges = self.edge_list.copy()
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
    
    def run_tomography(self):
        pass
    
    def gen_tomography_circuits(self):
        """
        """
        self.graphstate_circuit.barrier()
        circuits = {} # Dictionary of groups of circuits where batches are keys
        
        
        for batch, groups in self.batches.items():
            
            # Dictionary of circuits where measurement basis are keys
            batch_circuits = {}
            
            # Nx2 array of target (first two) edges
            targets = [g[:2] for g in groups]
            target_array = np.array(targets)
            flat_array = target_array.flatten()
            
            # List of adjacent qubits of all target edges in batch
            adj_qubit_list = []
            for target in targets:
                adj_qubit_list.extend(self.adj_qubits[target])
                
            
            circuitxx = self.graphstate_circuit.copy(batch + ' ' + 'XX')
            circuitxx.h(flat_array)
            batch_circuits['XX'] = circuitxx
            
            circuitxy = self.graphstate_circuit.copy(batch + ' ' + 'XY')
            circuitxy.sdg(target_array[:, 1].tolist())
            circuitxy.h(flat_array)
            batch_circuits['XY'] = circuitxy
            
            circuitxz = self.graphstate_circuit.copy(batch + ' ' + 'XZ')
            circuitxz.h(target_array[:, 0].tolist())
            batch_circuits['XZ'] = circuitxz
            
            circuityx = self.graphstate_circuit.copy(batch + ' ' + 'YX')
            circuityx.sdg(target_array[:, 0].tolist())
            circuityx.h(flat_array)
            batch_circuits['YX'] = circuityx
            
            circuityy = self.graphstate_circuit.copy(batch + ' ' + 'YY')
            circuityy.sdg(flat_array)
            circuityy.h(flat_array)
            batch_circuits['YY'] = circuityy
            
            circuityz = self.graphstate_circuit.copy(batch + ' ' + 'YZ')
            circuityz.sdg(target_array[:, 0].tolist())
            circuityz.h(target_array[:, 0].tolist())
            batch_circuits['YZ'] = circuityz
            
            circuitzx = self.graphstate_circuit.copy(batch + ' ' + 'ZX')
            circuitzx.h(target_array[:, 1].tolist())
            batch_circuits['ZX'] = circuitzx
            
            circuitzy = self.graphstate_circuit.copy(batch + ' ' + 'ZY')
            circuitzy.sdg(target_array[:, 1].tolist())
            circuitzy.h(target_array[:, 1].tolist())
            batch_circuits['ZY'] = circuitzy
            
            circuitzz = self.graphstate_circuit.copy(batch + ' ' + 'ZZ')
            batch_circuits['ZZ'] = circuitzz
            
            # Measure all target and adjacent qubits in batch
            for circuit in batch_circuits.values():
                circuit.measure(flat_array, flat_array) # Targets
                circuit.measure(adj_qubit_list, adj_qubit_list) # Adjacent
            
            circuits[batch] = batch_circuits
            
        self.circuits = circuits  
        
        return circuits
    
    def run_tomography_circuits(self, nshots=8192, sim=None):
        """
        """
        circuit_list = GraphState.circuits_tolist(self.circuits)
        self.nshots = nshots
        
        if sim is None:
            job = execute(circuit_list, backend=self.backend, shots=nshots)
        else:
            job = execute(circuit_list, backend=sim, shots=nshots)
        
        result = job.result()
        counts = self.from_result(result)
        #counts = result.get_counts()
        
        return counts
    
    def from_result(self, result): #FIX!!!!
        """"""
        name_list = self.gen_name_list()
        counts_dict = {batchn:{} for batchn in self.batches.keys()} #FIX!!!!
        for name in name_list:
            batchn, basis = name.split()
            counts_dict[batchn][basis] = result.get_counts(name)
        
        self.counts = counts_dict
        
        return counts_dict
    
    def gen_name_list(self):
        """"""
        name_list = []
        for i in range(len(self.batches)):
            for basis in basis_list:
                name_list.append(f'batch{i}' + ' ' + basis)
                
        return name_list
    
    def group_counts(self):
        """
        """
        n_rec = 1/self.nshots
        
        g_counts = {tuple(list(targ) + adj):{} for targ, adj in self.adj_qubits.items()}
        g_pvec = g_counts.copy()
        
        
        for group in g_counts.keys():            
            n = len(group)
                
            g_counts[group] = {basis:{bit_str:0 for bit_str in bit_str_list(n)} \
                        for basis in basis_list}
            g_pvec[group] = {basis:np.zeros(2**n) for basis in basis_list}
            
        for batchn, batch_counts in self.counts.items():
            
            for basis, counts in batch_counts.items():
                for bit_str, count in counts.items():
                    bit_str = bit_str[::-1] # Reverse bit string to align bit index
                    for group in self.batches[batchn]:
                        g_bit_str = splice_str(bit_str, list(group))
                        
                        g_counts[group][basis][g_bit_str] += count
                        g_pvec[group][basis][int(g_bit_str, 2)] += count*n_rec
            
        return g_counts, g_pvec
    
    def bucket_counts(self):
        """ 
        """
        b_counts = {edge:{} for edge in self.edge_list}
        for edge in b_counts.keys():
            n = len(self.adj_qubits[edge])
            b_counts[edge] = {bit_str:{basis:{bit:0 for bit in bit_str_list(2)} \
                                       for basis in basis_list} \
                                       for bit_str in bit_str_list(n)}
        
        for batchn, batch_counts in self.counts.items():
            batch_groups = self.batches[batchn]
            batch_targets = [g[:2] for g in batch_groups]
            for basis, counts in batch_counts.items():
                for bit_string, count in counts.items():
                    bit_string = bit_string[::-1]
                    for targ in batch_targets:
                        targ_str = splice_str(bit_string, list(targ))
                        adj_str = splice_str(bit_string, self.adj_qubits[targ])
                        
                        b_counts[targ][adj_str][basis][targ_str] += count
                
        self.b_counts = b_counts
                
        return b_counts
    
    def get_density_mat_dict(self):
        
        rho_dict = {edge:{} for edge in self.edge_list}
        for edge in rho_dict.keys():
            n = len(self.adj_qubits[edge])
            rho_dict[edge] = {bucket:None for bucket in bit_str_list(n)}
        
        try:
            b_counts = self.b_counts
        except:
            b_counts = self.bucket_counts()
        
        for edge, counts in b_counts.items():
            for bucket, basis_counts in counts.items():
                rho_dict[edge][bucket] = GraphState.recon_density_mat(basis_counts)
        
        self.rho_dict = rho_dict
        
        return rho_dict
    
    def get_negativities(self):
        """ 
        """
        N_all = self.rho_dict.copy()
        N_max = {}
        negativities = {}
        
        for edge, counts in self.rho_dict.items():
            N_sum = 0
            N_list = []
            for bucket, rho in counts.items():
                rho_pt = calc_ptrans(rho)
                N = calc_N(rho_pt)
                
                N_all[edge][bucket] = N
                N_sum += N
                N_list.append(N)
                
            negativities[edge] = N_sum/len(counts)
            N_max[edge] = max(N_list)
               
        self.N_all = N_all
        self.N_max = N_max
        self.negativities = negativities
        
        return negativities
    
# =============================================================================
#     def plot_negativities(self, mode='mean'):
#         """
#         """
#         if mode == 'mean':
#             negativities = self.negativities
#         elif mode == 'max':
#             negativities = self.N_max
#             
#         X = np.array([f'{edge[0]}-{edge[1]}' for edge in negativities.keys()])
#         Y = np.array(list(negativities.values()))
#         idx = Y.argsort()
#         X = X[idx]
#         Y = Y[idx]
#             
#         fig, ax = plt.subplots(figsize=(8, 6), dpi=144)
#         ax.scatter(X, Y)
#         
#         #ax.set_ylim(0, 0.5)
#         ax.set_yticks(np.arange(0, 0.55, 0.05))
#         ax.tick_params(axis='x', labelrotation=90)
#         ax.grid()
#         
#         ax.set_xlabel("Qubit Pairs")
#         ax.set_ylabel("Negativity")
# =============================================================================
    
        
    @staticmethod
    def circuits_tolist(circuit_dict):
        """"""
        circuit_list = []
        
        for batch in circuit_dict.values():
            for circuit in batch.values():
                circuit_list.append(circuit)
                
        return circuit_list
    
    @staticmethod
    def find_closest_physical(rho):
        """Find the closest physical density matrix with strictly positive eigenvalues"""
        
        rho = rho/rho.trace()
        rho_physical = np.zeros(rho.shape, dtype=complex)
        eigval, eigvec = la.eig(rho)
        
        idx = eigval.argsort()[::-1]
        
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigval_new = np.zeros(len(eigval), dtype=complex)
        
        i = len(eigval)
        a = 0
        
        while (eigval[i-1] + a/i) < 0:
            a += eigval[i-1]
            i -= 1
            
        for j in range(i):
            eigval_new[j] = eigval[j] + a/i
            rho_physical += eigval_new[j] * np.outer(eigvec[:, j], eigvec[:, j].conjugate())
        
        return rho_physical
    
    @staticmethod
    def recon_density_mat(basis_counts):
        """"""
        
        S_dict = GraphState.calc_S_dict(basis_counts)
        rho = np.zeros([4, 4], dtype=complex)
        N = 0.25
        for basis, S in S_dict.items():
            rho += N*S*pauli_n(basis)
        
        rho = GraphState.find_closest_physical(rho)
        
        return rho
    
    @staticmethod
    def calc_S_dict(basis_counts):
        """
        """
        
        S_dict = {basis:0 for basis in ext_basis_list}
        S_dict['II'] = 1.
        
        for basis, counts in basis_counts.items():
            
            n = sum(counts.values())
            
            S = counts['00'] - counts['01'] - counts['10'] + counts['11']
            S_dict[basis] = S/n
            
            if basis[0] == basis[1]:
                S_IX = counts['00'] - counts['01'] + counts['10'] - counts['11']
                S_XI = counts['00'] + counts['01'] - counts['10'] - counts['11']
                S_dict['I' + basis[1]] = S_IX/n
                S_dict[basis[0] + 'I'] = S_XI/n
        
        return S_dict
    

    
class GHZState(System):
    """
    """
    def __init__(self, backend):
        # Initialise from input backend
        super().__init__(backend)
            
    def gen_pathtree(self, source, nodes):
        """
        """
        length = {q: self.nqubits for q in range(self.nqubits)}
        path = {q: [] for q in range(self.nqubits)}
        degree = {q: 0 for q in range(self.nqubits)}
        tlength = 0
        
        length[source] = 0
        path[source] = [source]
        
        visited = []
        unvisited = length.copy()
        
        for i in range(nodes):
            u = min(unvisited, key=unvisited.get)
            visited.append(u)
            del unvisited[u]
            try:
                tlength += self.edge_params[tuple(sorted((u, path[u][-2])))]
                degree[path[u][-2]] += 1
            except: pass
            
            for v in self.connections[u]:
                alt = len(path[u]) - 1 + self.edge_params[tuple(sorted((u, v)))]
                if alt < length[v]:
                    unvisited[v] = length[v] = alt
                    path[v] = path[u] + [v]
        
            pathtree = {q: (path[q], sum(list(map(degree.get, path[q][:-1])))) for q in visited}
                
        return pathtree, tlength
    
    def gen_circuit(self, nodes):
        """"""
        minlength = 100.
        minq = 0
        mintree = {}
        for q in range(self.nqubits):
            pathtree, tlength = self.gen_pathtree(q, nodes)
            if tlength < minlength:
                minlength = tlength
                minq = q
                mintree = pathtree
                
        return mintree
    
    
if __name__ == "__main__":
    startup()
    provider = IBMQ.get_provider("ibm-q-melbourne")
    backend = provider.get_backend("ibm_perth")
    #backend = provider.get_backend("ibm_cairo")
    #backend = provider.get_backend("ibmq_brooklyn")
    #backend = provider.get_backend("ibm_washington")
    
    print('\033[4mTime Stamps:\033[0m')
    time0 = time()
    test = GraphState(backend)
    time1 = time()
    print(f'Class Initialisation: {time1 - time0}')
    circuits = test.gen_tomography_circuits()
    
    time2 = time()
    print(f'Generating Circuits: {time2 - time1}')
    sim = Aer.get_backend('aer_simulator')
    counts = test.run_tomography_circuits(8192)
    #counts = test.run_tomography_circuits(8192, sim)
    
    #job = backend.jobs()[0]
    #counts = test.from_result(job.result())
    
    time3 = time()
    print(f'Running Circuits: {time3 - time2}')
    b_counts = test.bucket_counts()
    
    time4 = time()
    print(f'Bucketing Counts: {time4 - time3}')
    rho_dict = test.get_density_mat_dict()
    
    time5 = time()
    print(f'Reconstructing Density Matrices: {time5 - time4}')
    negativities = test.get_negativities()
    
# =============================================================================
#     print('\033[4mTime Stamps:\033[0m')
#     print(f'Class Initialisation: {time1 - time0}')
#     print(f'Generating Circuits: {time2 - time1}')
#     print(f'Running Circuits: {time3 - time2}')
#     print(f'Bucketing Counts: {time4 - time3}')
#     print(f'Reconstructing Density Matrices: {time5 - time4}')
# =============================================================================
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:01:06 2022

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
import numpy.linalg as la

# Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, IBMQ, Aer, execute
from qiskit.providers.aer.noise import NoiseModel

# Local modules
from utilities import startup, pauli_n, bit_str_list
from entanglebase import EntangleBase

# Useful variables
basis_list = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
ext_basis_list = ['II', 'IX', 'IY', 'IZ',
                  'XI', 'XX', 'XY', 'XZ',
                  'YI', 'YX', 'YY', 'YZ',
                  'ZI', 'ZX', 'ZY', 'ZZ']

class GraphState(EntangleBase):
    """
    """
    
    def __init__(self, backend):
        """
        

        Parameters
        ----------
        backend : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__(backend)
        
        self.adj_qubits = self.__get_adjs()
        self.circuit = self.__gen_graphstate_circuit()
    
    def __get_adjs(self):
        """
        Get the edge-qubit adjacencies for every edge in the quantum device

        Returns
        -------
        adj_qubits : TYPE
            DESCRIPTION.

        """
        
        # adj_edges = {}
        adj_qubits = {}
        # Iterate over every edge
        for edge in self.edge_list:
            other_edges = self.edge_list.copy()
            other_edges.remove(edge)
            #connected_edges = []
            connected_qubits = []
            # Iterate over all other edges
            for edgej in other_edges:
                if np.any(np.isin(edge, edgej)):
                    #connected_edges.append(edgej)
                    for q in edgej:
                        if q not in edge: connected_qubits.append(q)
            #adj_edges[edge] = connected_edges
            adj_qubits[edge] = connected_qubits
            
        return adj_qubits
    
    def __gen_graphstate_circuit(self):
        """
        Generate a graph state circuit over every physical edge

        Returns
        -------
        circ : TYPE
            DESCRIPTION.

        """
             
        circ = QuantumCircuit(self.nqubits)
        unconnected_edges = self.edge_list.copy()
        # Apply Hadamard gates to every qubit
        circ.h(list(range(self.nqubits)))
        # Connect every edge with cz gates
        while unconnected_edges:
            connected_qubits = [] # Qubits already connected in the current time step
            remove = []
            for edge in unconnected_edges:
                if np.any(np.isin(edge, connected_qubits)) == False:
                    circ.cz(edge[0], edge[1])
                    connected_qubits.extend(edge)
                    remove.append(edge)
            # Remove connected edges from unconnected edges list
            for edge in remove:
                unconnected_edges.remove(edge)   
                
        return circ
    
    def run_qst(self, qrem=False, sim=None, shots=8192):
        """
        

        Parameters
        ----------
        sim : TYPE, optional
            DESCRIPTION. The default is False.
        qrem : TYPE, optional
            DESCRIPTION. The default is False.
        nshots : TYPE, optional
            DESCRIPTION. The default is 8192.

        Returns
        -------
        rho_dict : TYPE
            DESCRIPTION.

        """
        
        self.gen_qst_circuits()
        self.run_qst_circuits(qrem, sim, shots)
        
        rho_dict = self.recon_density_mats()
        
        return rho_dict
    
    def qst_from_result(self, result):
        
        self.gen_qst_circuits()
        self.counts_from_result(result)
        
        rho_dict = self.recon_density_mats()
        
        return rho_dict
    
    def gen_batches(self):
        """
        Get a dictionary of tomography batches, where keys are batch numbers
        and values are lists of tomography groups (targets + adj qubits)

        Returns
        -------
        batches : dict
            DESCRIPTION.

        """

        batches = {}
        group_list = []
        
        unbatched_edges = self.edge_list.copy()
        i = 0
        # Loop over unbatched edges until no unbatched edges remain
        while unbatched_edges:
            batches[f'batch{i}'] = []
            batched_qubits = []
            remove = []
            
            for edge in unbatched_edges:
                group = tuple(list(edge) + self.adj_qubits[edge])
                # Append edge to batch only if target and adjacent qubits have 
                # not been batched in the current cycle
                if np.any(np.isin(group, batched_qubits)) == False:
                    batches[f'batch{i}'].append(group)
                    group_list.append(group)
                    
                    batched_qubits.extend(group)
                    remove.append(edge)
                    
            for edge in remove:
                unbatched_edges.remove(edge)
            i += 1
            
            self.batches = batches
            self.group_list = group_list
            
        return batches
    
    def gen_qst_circuits(self):
        """
        Generates circuits for quantum state tomography

        Returns
        -------
        circuits : dict
            DESCRIPTION.

        """
        # Generate batches of groups (target edges + adjacent qubits) to perform
        # QST in parallel
        try:
            batches = self.batches
        except AttributeError:
            batches = self.gen_batches()
        
        circuits = {} # Dictionary of groups of circuits where batches are keys
        name_list = [] # List of circuit names
        
        graphstate = self.circuit.copy()
        graphstate.barrier()
        
        for batch, groups in batches.items():
            
            # Dictionary of circuits where measurement basis are keys
            batch_circuits = {}
            
            # Nx2 array of target (first two) edges
            targets = [g[:2] for g in groups]
            targ_array = np.array(targets)
            flat_array = targ_array.flatten()
            
            # Create circuits for each basis combination over target pairs
            circxx = graphstate.copy(batch + ' ' + 'XX')
            circxx.h(flat_array)
            batch_circuits['XX'] = circxx
            
            circxy = graphstate.copy(batch + ' ' + 'XY')
            circxy.sdg(targ_array[:, 1].tolist())
            circxy.h(flat_array)
            batch_circuits['XY'] = circxy
            
            circxz = graphstate.copy(batch + ' ' + 'XZ')
            circxz.h(targ_array[:, 0].tolist())
            batch_circuits['XZ'] = circxz
            
            circyx = graphstate.copy(batch + ' ' + 'YX')
            circyx.sdg(targ_array[:, 0].tolist())
            circyx.h(flat_array)
            batch_circuits['YX'] = circyx
            
            circyy = graphstate.copy(batch + ' ' + 'YY')
            circyy.sdg(flat_array)
            circyy.h(flat_array)
            batch_circuits['YY'] = circyy
            
            circyz = graphstate.copy(batch + ' ' + 'YZ')
            circyz.sdg(targ_array[:, 0].tolist())
            circyz.h(targ_array[:, 0].tolist())
            batch_circuits['YZ'] = circyz
            
            circzx = graphstate.copy(batch + ' ' + 'ZX')
            circzx.h(targ_array[:, 1].tolist())
            batch_circuits['ZX'] = circzx
            
            circzy = graphstate.copy(batch + ' ' + 'ZY')
            circzy.sdg(targ_array[:, 1].tolist())
            circzy.h(targ_array[:, 1].tolist())
            batch_circuits['ZY'] = circzy
            
            circzz = graphstate.copy(batch + ' ' + 'ZZ')
            batch_circuits['ZZ'] = circzz
            
            
            for circ in batch_circuits.values():
                name_list.append(circ.name)
                # Create a seperate classical register for each group in batch
                # and apply measurement gates respectively
                for group in groups:
                    cr = ClassicalRegister(len(group))
                    circ.add_register(cr)
                    circ.measure(group, cr)
            
            circuits[batch] = batch_circuits
            
            self.qst_circuits = circuits
            self.name_list = name_list
                    
        return circuits
    
    def run_qst_circuits(self, qrem=False, sim=None, shots=8192):
        """
        Runs the quantum state tomography circuits to obtain raw counts as dict

        Parameters
        ----------
        sim : bool, optional
            DESCRIPTION. The default is False.
        shots : int, optional
            Number of repetitions per QST circuit. The default is 8192.

        Returns
        -------
        counts : dict
            DESCRIPTION.

        """
        self.qrem = qrem
        self.sim = sim
        self.shots = shots
        
        # Convert circuits dict into list form
        circ_list = []
        for batch in self.qst_circuits.values():
            for circuit in batch.values():
                circ_list.append(circuit)
        
        # Generate QREM circuits and append to circ_list if qrem == True
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
            circ_list.extend(qrem_circuits)
        
        # If circuits are executed on a simulator or real backend
        if sim is None:
            result = execute(circ_list, backend=self.backend, 
                             shots=shots).result()
        elif sim == "ideal":
            result = execute(circ_list, Aer.get_backend('aer_simulator'), 
                             shots=shots).result()
        elif sim == "device":
            # Obtain device and noise model parameters
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates
            
            result = execute(circ_list, Aer.get_backend('aer_simulator'),
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model, 
                          shots=shots).result()
            
        # Get raw counts as dict
        counts = self.counts_from_result(result, qrem)
        
        return counts
    
    def counts_from_result(self, result):
        """
        Load raw counts from qiskit result converting into dict form

        Parameters
        ----------
        result : qiskit.Result
            DESCRIPTION.

        Returns
        -------
        counts : dict
            DESCRIPTION.

        """
        
        counts = {batch:{} for batch in self.batches.keys()}
        
        try: # If function is called from self.run_qst_circuits()
            self.shots
        except AttributeError: # If function is called directly
            counts0 = result.get_counts(self.name_list[0])
            self.shots = sum(counts0.values())
            try: # Try to obtain QREM results
                result.get_counts('qrem0')
                self.qrem = True
            except:
                self.qrem = False
        
        # Load counts as dict
        for name in self.name_list:
            batch, basis = name.split()
            counts[batch][basis] = result.get_counts(name)
            
        self.qst_counts = counts
        
        if self.qrem is True:
            qrem_counts = [result.get_counts('qrem0'), 
                           result.get_counts('qrem1')]
            
            M_list = [np.zeros((2, 2)) for i in range(self.nqubits)]
            for jj, counts in enumerate(qrem_counts):
                for bit_str, count in counts.items():
                    for i, q in enumerate(bit_str[::-1]):
                        ii = int(q)
                        M_list[i][ii, jj] += count
                        
            # Normalise
            norm = 1/self.shots
            for M in M_list:
                M *= norm
                
            self.M_list = M_list
        
        return counts
    
    def group_counts(self):
        """
        Splits (raw) batch counts into counts for each group (target + 
        neighbouring qubits as dictionary of bit string counts and dictionary
        of probability vectors

        Returns
        -------
        g_counts : dict
            Group (bit string) counts.
        g_vecs : dict
            Group probability vectors.

        """
        
        # Construct dictionary keys
        g_counts = {}
        g_vecs = {}
        for group in self.group_list:            
            n = len(group)
            g_counts[group] = {basis:{bit_str:0.
                                      for bit_str in bit_str_list(n)} \
                               for basis in basis_list}
            g_vecs[group] = {basis:np.zeros(2**n) for basis in basis_list}
        
        # Nested loop over each bit string over each basis over each batch in 
        # raw counts obtained from self.counts_from_result()
        for batch, batch_counts in self.qst_counts.items():
            for basis, counts in batch_counts.items():
                for bit_str, count in counts.items():
                    # Reverse and split bit string key in counts
                    split = bit_str[::-1].split()
                    # Loop over every group in batch and increment corresponding 
                    # bit string counts
                    for i, group in enumerate(self.batches[batch]):
                        g_counts[group][basis][split[i]] += count
                        g_vecs[group][basis][int(split[i], 2)] += count
        
        return g_counts, g_vecs
            
    def bin_pvecs(self, g_counts):
        """
        

        Returns
        -------
        b_pvecs : TYPE
            DESCRIPTION.

        """
        
        b_pvecs = {}
        for edge in self.edge_list:
            n = len(self.adj_qubits[edge])
            b_pvecs[edge] = {bn:{basis:np.zeros(4) 
                                 for basis in basis_list} 
                             for bn in bit_str_list(n)}
            
        for group, basis_counts in g_counts.items():
            edge = group[:2]
            for basis, counts in basis_counts.items():
                for bit_str, count in counts.items():
                    idx = int(bit_str[:2], 2)
                    bn = bit_str[2:]
                    
                    b_pvecs[edge][bn][basis][idx] += count
                  
                # Normalise
                for bn in bit_str_list(len(self.adj_qubits[edge])):
                    pvec = b_pvecs[edge][bn][basis]
                    norm = 1/pvec.sum()
                    b_pvecs[edge][bn][basis] = pvec*norm
                 
        return b_pvecs
    
    def recon_density_mats(self):
        """
        

        Returns
        -------
        rho_dict : TYPE
            DESCRIPTION.

        """
        rho_dict = {edge:{} for edge in self.edge_list}
        
        g_counts, g_vecs = self.group_counts()
        b_pvecs = self.bin_pvecs(g_counts)
        
        for edge, bns in b_pvecs.items():
            for bn, pvecs in bns.items():
                rho_dict[edge][bn] = GraphState.calc_rho(pvecs)
                
        self.rho_dict = rho_dict
        
        # If QREM is applied, additionally reconstruct mitigated density
        # matrices without deleting unmitigated matrices
        if self.qrem is True:
            rho_dict_mit = {edge:{} for edge in self.edge_list}
            g_counts_mit, g_vecs_mit = self.apply_qrem(g_counts, g_vecs)
            b_pvecs_mit = self.bin_pvecs(g_counts_mit)
            
            for edge, bns in b_pvecs_mit.items():
                for bn, pvecs in bns.items():
                    rho_dict_mit[edge][bn] = GraphState.calc_rho(pvecs)
                    
            self.rho_dict_mit = rho_dict_mit
            
            return rho_dict_mit
                
        return rho_dict
    
    def get_negativities(self, mit=True):
        """
        

        Parameters
        ----------
        mit : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        n_mean : TYPE
            DESCRIPTION.
        n_all : TYPE
            DESCRIPTION.

        """
        
        if mit and self.qrem:
            rho_dict = self.rho_dict_mit.copy()
        else:
            rho_dict = self.rho_dict.copy()
            
        n_all = {edge:{} for edge in self.edge_list}
        n_mean = {}
        
        for edge, bns in rho_dict.items():
            n_sum = 0.
            n_list = []
            
            for bn, rho in bns.items():
                n = GraphState.calc_n(rho)
                
                n_all[edge][bn] = n
                n_sum += n
                n_list.append(n)
                
            n_mean[edge] = n_sum/len(bns)
            
        return n_mean, n_all
    
    def gen_qrem_circuits(self):
        """"""     

        circ0 = QuantumCircuit(self.nqubits, name='qrem0')
        circ0.measure_all()
        
        circ1 = QuantumCircuit(self.nqubits, name='qrem1')
        circ1.x(range(self.nqubits))
        circ1.measure_all()
        
        self.qrem_circuits = [circ0, circ1]
        
        return [circ0, circ1]
    
    def apply_qrem(self, g_counts, g_vecs):
        
        g_counts_mit = g_counts.copy()
        g_vecs_mit = g_vecs.copy()
        
        for group, vecs in g_vecs.items():
            n = len(group)
            M_inv = la.inv(self.calc_M_multi(group))
            for basis, vec in vecs.items():
                vec_mit = np.matmul(M_inv, vec)
                g_vecs_mit[group][basis] = vec_mit
                
                for i, count in enumerate(vec_mit):
                    bit_str = bin(i)[2:].zfill(n)
                    g_counts_mit[group][basis][bit_str] = count
                    
        return g_counts_mit, g_vecs_mit
            
    def calc_M_multi(self, qubits):
        
        M = self.M_list[qubits[0]]
        for q in qubits[1:]:
            M_new = np.kron(M, self.M_list[q])
            M = M_new
            
        return M
    
    @staticmethod 
    def calc_rho(pvecs):
        """
        

        Parameters
        ----------
        pvecs : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.

        """
        
        rho = np.zeros([4, 4], dtype=complex)
        
        # First calculate the Stokes parameters s
        s_dict = {basis:0. for basis in ext_basis_list}
        s_dict['II'] = 1. # S for 'II' always equals 1
        
        # Calculate s in each (extended) basis
        for basis, pvec in pvecs.items():
            # s for basis not containing I
            s_dict[basis] = pvec[0] - pvec[1] - pvec[2] + pvec[3]
            # s for basis 'IX' and 'XI' can be derived from 'XX' etc..
            if basis[0] == basis[1]:
                s_dict['I' + basis[0]] = pvec[0] - pvec[1] + pvec[2] - pvec[3]
                s_dict[basis[0] + 'I'] = pvec[0] + pvec[1] - pvec[2] - pvec[3]
        
        # Weighted sum of basis matrices
        for basis, s in s_dict.items():
            rho += 0.25*s*pauli_n(basis)
            
        # Convert raw density matrix into closest physical density matrix using
        # Smolin's algorithm (2011)
        rho = GraphState.find_closest_physical(rho)
                
        return rho
    
    @staticmethod
    def find_closest_physical(rho):
        """
        

        Parameters
        ----------
        rho : TYPE
            DESCRIPTION.

        Returns
        -------
        rho_physical : TYPE
            DESCRIPTION.

        """
        
        rho = rho/rho.trace()
        rho_physical = np.zeros(rho.shape, dtype=complex)
        # Step 1: Calculate eigenvalues and eigenvectors
        eigval, eigvec = la.eig(rho)
        # Rearranging eigenvalues from largest to smallest
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigval_new = np.zeros(len(eigval), dtype=complex)
        
        # Step 2: Let i = number of eigenvalues and set accumulator a = 0
        i = len(eigval)
        a = 0
        
        while (eigval[i-1] + a/i) < 0:
            a += eigval[i-1]
            i -= 1
            
        # Step 4: Increment eigenvalue[j] by a/i for all j <= i
        for j in range(i):
            eigval_new[j] = eigval[j] + a/i
            # Step 5 Construct new density matrix
            rho_physical += eigval_new[j] * np.outer(eigvec[:, j], eigvec[:, j].conjugate())
        
        return rho_physical
    
    @staticmethod
    def calc_n(rho):
        """
        Calculate the negativity of entanglement given a 2-qubit density matrix

        Parameters
        ----------
        rho : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        rho_pt = GraphState.ptrans(rho)
        w, v = la.eig(rho_pt)
        n = np.sum(w[w<0])
        
        return abs(n)
    
    @staticmethod
    def ptrans(rho):
        """Obtain the partial transpose of a 4x4 array (A kron B) w.r.t B"""
        rho_pt = np.zeros(rho.shape, dtype=complex)
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                rho_pt[i:i+2, j:j+2] = rho[i:i+2, j:j+2].transpose()
        
        return rho_pt

        
        
    
if __name__ == "__main__":
    startup()
    provider = IBMQ.get_provider("ibm-q-melbourne")
    #backend = provider.get_backend("ibm_perth")
    backend = provider.get_backend("ibmq_montreal")
    #backend = provider.get_backend("ibm_brooklyn")
    #backend = provider.get_backend("ibm_washington")
    
    test = GraphState(backend)
    
    time0 = time()
    #test.run_qst(qrem=True, sim=None, shots=8192)
    
    job = backend.jobs()[0]
    result = job.result()
    time1 = time()
    
    rho_dict = test.qst_from_result(result)
    time2 = time()
    
    negativities, _ = test.get_negativities()
    print("\033[4mNegativities\033[0m")
    for n in negativities.values(): print(n)
    time3 = time()
    
    print("")
    print("\033[4mTime Stamps\033[0m")
    #print(f'Running Circuits: {time1 - time0}')
    print(f"Fetching Results: {time1 - time0}")
    print(f"Reconstructing Density Matrices: {time2 - time1}")
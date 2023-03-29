# -*- coding: utf-8 -*-
"""
Created on Thu May 12 00:04:24 2022

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# Qiskit libraries
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit.visualization import timeline_drawer

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeCairoV2

from qiskit.visualization import plot_histogram
import mthree

# Local modules
from entanglebase import EntangleBase
from utilities import startup, run_cal, load_cal


class GHZState(EntangleBase):

    def __init__(self, backend):

        super().__init__(backend)


        self.ghz_circuit = None
        self.initial_layout = []
        self.delays = None

        self.shots = None

    def gen_ghz_circuit(self, nodes, source=None, output_error=False):

        circ = QuantumCircuit(nodes)

        # If source is None
        if source is None:
            source, cx_instr, initial_layout, depth, terror_dict = self.find_opt_source(
                nodes)
        # If source qubit is specified
        else:
            cx_instr, initial_layout, depth, terror_dict = self.gen_circ_instr(
                nodes, source)

        # Construct circuit in Qiskit
        circ.h(0)
        for edge in cx_instr:
            circ.cx(*edge)

        self.ghz_circuit = circ
        self.ghz_size = nodes
        self.initial_layout = initial_layout

        if output_error is True:
            return circ, initial_layout, terror_dict

        return circ, initial_layout
    
    def ghz_circ_from_instr(self, instr):
        
        initial_layout = list(dict.fromkeys([q for edge in instr for q in edge]))
        nodes = len(initial_layout)
        cx_instr = [(initial_layout.index(a), initial_layout.index(b)) for (a, b) in instr]
        
        # Construct circuit in Qiskit
        circ = QuantumCircuit(nodes)
        circ.h(0)
        for edge in cx_instr:
            circ.cx(*edge)
        
        self.ghz_circuit = circ
        self.ghz_size = nodes
        self.initial_layout = initial_layout
        
        return circ, initial_layout
        

    def find_opt_source(self, nodes, error_key='cumcx'):

        cx_instr = None
        initial_layout = None
        min_depth = self.nqubits
        min_terror = self.nqubits*1000000

        for q in range(self.nqubits):
            instr, layout, depth, terror_dict = self.gen_circ_instr(nodes, q)
            # If CNOT depth is less than or equal to
            if depth <= min_depth:
                # If total error is less than
                if terror_dict[error_key] < min_terror or depth < min_depth:
                    source = q
                    cx_instr = instr
                    initial_layout = layout
                    min_depth = depth
                    min_terror = terror_dict[error_key]

        return source, cx_instr, initial_layout, min_depth, terror_dict

    def gen_circ_instr(self, nodes, source, mapped=False):
        """Modified Dijkstra's algorithm"""

        terror_dict = {'cumcx': 0,
                       'meancx': 0,
                       't1': 0,
                       't2': 0}

        length = {q: self.nqubits for q in range(self.nqubits)}
        next_depth = {}
        path = {q: [] for q in range(self.nqubits)}
        #degree_visited = {q: 0 for q in range(self.nqubits)}
        degree = dict(self.graph.degree)
        degree_unvisited = degree.copy()
        error = {q: 1 for q in range(self.nqubits)}

        length[source] = 0
        path[source] = [source]
        error[source] = 0

        visited = []
        unvisited = length.copy()

        for i in range(nodes):
            # Find minimum length (CNOT depth) unconnected nodes
            lmin = min(unvisited.values())
            #unvisited_lmin = {key: value for key, value in unvisited.items() if value == lmin}
            u_lmin = [key for key, value in unvisited.items() if value == lmin]
            # Pick potential nodes with largest degree
            degree_lmin = {key: degree_unvisited[key] for key in u_lmin}
            dmax = max(degree_lmin.values())
            u_lmin_dmax = [key for key,
                           value in degree_lmin.items() if value == dmax]
            # Pick node with lowest propagated CNOT error
            u = min(u_lmin_dmax, key=error.get)
            degree_unvisited[u] -= 1
            # Update error dict
            terror_dict['cumcx'] += error[u]
            terror_dict['t1'] += self.backend.properties().t1(u)
            terror_dict['t2'] += self.backend.properties().t2(u)

            visited.append(u)
            del unvisited[u]
            next_depth[u] = length[u] + 1
            

            for v in self.connections[u]:
                degree_unvisited[v] -= 1
                alt = length[u] + 1
                if alt < length[v]:
                    unvisited[v] = length[v] = alt
                    path[v] = path[u] + [v]
                    error[v] = sum_error(
                        error[u], self.edge_params[tuple(sorted((u, v)))])
                    
            try:
                u_prev = path[u][-2]
                next_depth[u_prev] = length[u] + 1
                #print(next_depth)
                
                for v_prev in self.connections[u_prev]:
                    if v_prev in unvisited:
                        connected = set(self.connections[v_prev]) - set(unvisited)
                        length[v_prev] = unvisited[v_prev] = next_depth[min(connected, key=next_depth.get)]
            except: pass

        cx_instr = []
        
        if mapped is True:
            qubits = []
            edges = []
            depths = []
            for q in visited[1:]:
                qubits.append(q)
                edges.append(path[q][-2:])
                depths.append(length[q])
                
            return qubits, edges, depths
                
        
        for q in visited[1:]:
            c, t = path[q][-2:]
            cx_instr.append((visited.index(c), visited.index(t)))
        initial_layout = visited
        depth = length[u]   

        return cx_instr, initial_layout, depth, terror_dict

    def gen_fid_circuits(self, delays=[0], dynamic_decoupling=False, pi_pulse=True, pulses=4):

        self.delays = delays
        try:
            self.dt = delays[1] - delays[0]
        except:
            pass

        circ_list = []

        ghz_prep = self.ghz_circuit.copy()
        ghz_prep.barrier()
        ghz_inv = ghz_prep.inverse()

        # MQC circuit parameters
        phi_step = np.pi/(self.ghz_size+1)
        phi_max = phi_step*(2*self.ghz_size+1)
        phi_list = np.linspace(0, phi_max, 2*self.ghz_size+2).tolist()

        for i, t in enumerate(delays):
            # Circuit to measure population
            circ_pop = ghz_prep.copy(f'pop_t{i}')
            if t == 0:  # For 0 delay
                if pi_pulse is True:
                    circ_pop.x(range(self.ghz_size))
            else:
                if dynamic_decoupling is True:
                    circ_pop.compose(self.gen_circ_dd(t, pulses), inplace=True)
                else:
                    circ_pop.delay(t)
            circ_pop.measure_all()
            circ_list.append(circ_pop)

            # Circuit to measure coherence
            mqc_base = ghz_prep.copy()
            if t == 0:  # For 0 delay
                if pi_pulse is True:
                    mqc_base.x(range(self.ghz_size))
            else:
                if dynamic_decoupling is True:
                    mqc_base.compose(self.gen_circ_dd(t, pulses), inplace=True)
                else:
                    thalf = self.format_delays(t/2, unit='dt')
                    mqc_base.delay(thalf)
                    if pi_pulse is True:
                        mqc_base.x(range(self.ghz_size))
                    mqc_base.delay(thalf)

            for j, phi in enumerate(phi_list):
                circ_mqc = mqc_base.copy(f'mqc_phi{j}_t{i}')
                if j == 0:
                    pass
                else:
                    circ_mqc.rz(phi, range(self.ghz_size))
                circ_mqc.compose(ghz_inv, inplace=True)
                circ_mqc.measure_all()
                circ_list.append(circ_mqc)

        self.fid_circuits = circ_list

        return circ_list

    def gen_circ_dd(self, t, pulses):

        circ_dd = QuantumCircuit(self.ghz_size)

        tpulses = int(pulses*t/self.dt)
        tdelay = t - tpulses*self.tx

        spacings = self.format_delays(
            [tdelay/tpulses]*(tpulses - 1), unit='dt')
        padding = self.format_delays(0.5*(tdelay - spacings.sum()), unit='dt')

        circ_dd.delay(padding)
        for t in spacings:
            circ_dd.x(range(self.ghz_size))
            circ_dd.delay(t)
        circ_dd.x(range(self.ghz_size))
        circ_dd.delay(padding)

        return circ_dd

    def format_delays(self, delays, unit='ns'):

        try:
            # For array of times
            n = len(delays)
        except TypeError:
            n = None

        # Convert delays based on input unit
        dt = self.backend.configuration().dt
        if unit == 'ns':
            scale = 1e-9/dt
        elif unit == 'us':
            scale = 1e-6/dt
        elif unit == 'dt':
            scale = 1

        # Qiskit only accepts multiples of 16*dt
        if n is None:
            # For single time
            delays_new = np.floor(delays*scale/16)*16
        else:
            # For array of times
            delays_new = np.zeros(n)
            for i, t in enumerate(delays):
                delays_new[i] = np.round(t*scale/16)*16

        return delays_new

    def run_circuits(self, shots=8192, sim=False, printid=True):

        self.shots = shots

        if sim is True:
            sim = Aer.get_backend('aer_simulator')
            noise_model = NoiseModel.from_backend(self.backend)
            coupling_map = self.backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            job = execute(self.fid_circuits, backend=sim,
                          initial_layout=self.initial_layout,
                          coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          noise_model=noise_model,
                          shots=shots)

        else:
            job = execute(self.fid_circuits, backend=self.backend,
                          initial_layout=self.initial_layout,
                          shots=shots)

            if printid is True:
                print('Job ID:', job.job_id())

        return job

    def counts_from_result(self, result, initial_layout=None, delays=None):

        if self.shots is None:
            self.shots = result.results[0].shots

        if initial_layout is not None:
            self.initial_layout = initial_layout
            self.ghz_size = len(initial_layout)

        pop_counts = {}
        mqc_counts = {}

        for i in range(len(self.delays)):
            ti = f't{i}'
            pop_counts[ti] = result.get_counts('pop_' + ti)

            mqc_counts[ti] = []
            for j in range(2*self.ghz_size+2):
                phij = f'phi{j}'
                mqc_counts[ti].append(
                    result.get_counts('mqc_' + phij + '_' + ti))

        return pop_counts, mqc_counts

    def load_cal(self, mit=None):
        if mit is None:
            mit = load_cal(self.backend)
        self.mit = mit

        return mit

    def mitigate_counts(self, pop_counts, mqc_counts):

        pop_counts_mit = pop_counts.copy()
        mqc_counts_mit = mqc_counts.copy()

        for t, counts in pop_counts.items():
            counts_mit = self.mit.apply_correction(counts, self.initial_layout)
            counts_mit = counts_mit.nearest_probability_distribution()
            # Un-normalize
            for bitstr, count in counts_mit.items():
                counts_mit[bitstr] = count*self.shots
            pop_counts_mit[t] = counts_mit

        for t, counts in mqc_counts.items():
            counts_mit = self.mit.apply_correction(counts, self.initial_layout)
            counts_mit = counts_mit.nearest_probability_distribution()
            # Un-normalize
            for i, counts_dict in enumerate(counts_mit):
                for bitstr, count in counts_dict.items():
                    counts_mit[i][bitstr] = count*self.shots

            mqc_counts_mit[t] = counts_mit

        return pop_counts_mit, mqc_counts_mit

    def calc_fidelity(self, pop_counts, mqc_counts):

        fidel = dict.fromkeys(pop_counts)

        pop = self.calc_pop(pop_counts)
        coh = self.calc_coh(mqc_counts)

        if len(fidel) == 1:
            fidel = (pop + coh)/2
        else:
            for t in fidel.keys():
                fidel[t] = (pop[t] + coh[t])/2

        return fidel, pop, coh

    def calc_pop(self, pop_counts):

        pop = dict.fromkeys(pop_counts)

        for t, counts in pop_counts.items():
            excited = counts.get('1'*self.ghz_size, 0)
            ground = counts.get('0'*self.ghz_size, 0)
            pop[t] = (excited + ground)/self.shots

        if len(pop) == 1:
            pop = pop[t]

        return pop

    def calc_coh(self, mqc_counts, plot=False):

        coh = dict.fromkeys(mqc_counts)

        for t, counts_list in mqc_counts.items():
            S_phi = [counts.get('0'*self.ghz_size, 0)/self.shots
                     for counts in counts_list]

            S_ifft = fft.ifft(S_phi)
            freqs = fft.fftfreq(len(S_phi)) * (2*self.ghz_size + 2)
            freqs = np.rint(freqs).astype(int).tolist()

            I_n = np.abs(S_ifft[freqs.index(self.ghz_size)])
            C = 2*np.sqrt(I_n)

            coh[t] = C

        if len(coh) == 1:
            coh = coh[t]

        return coh


def sum_error(a, b):
    return a + b - a*b

def lim_array(array):
    n_prev = 1
    for i, n in enumerate(array):
        if n > n_prev:
            array[i:] = np.zeros(len(array[i:]))
            break
        n_prev = n
        
    return array


if __name__ == "__main__":
    provider = startup()
    backend = provider.get_backend('ibm_washington')

    test = GHZState(backend)
    cx_instr, initial_layout, max_depth, tlength = test.gen_circ_instr(127, 63)

    #ghz_circ, initial_layout = test.gen_ghz_circuit(20)
    """
    delays = list(range(0, 8800, 800))
    delays = test.format_delays(delays, unit='ns')

    circ_list1 = test.gen_fid_circuits(delays, dynamic_decoupling=False)
    job1 = test.run_circuits(sim=False)

    circ_list2 = test.gen_fid_circuits(delays, dynamic_decoupling=True)
    job2 = test.run_circuits(sim=False)

    result1 = job1.result()
    pop_counts1, mqc_counts1 = test.counts_from_result(result1)
    Y1 = list(test.calc_coh(mqc_counts1).values())

    result2 = job2.result()
    pop_counts2, mqc_counts2 = test.counts_from_result(result2)
    Y2 = list(test.calc_coh(mqc_counts2).values())

    X = list(range(len(Y1)))

    plt.plot(X, Y1, label='No DD')
    plt.plot(X, Y2, label='DD')
    plt.xlabel('Time (us)')
    plt.ylabel('Coherence')
    plt.legend()
    """

    # mit.cals_from_matrices(test.M_list)
    #test_counts = counts[1]['t0'][0]
    #mit_counts = mit.apply_correction(test_counts, test.initial_layout)
    #mit_counts = {key: value*test.shots for key, value in mit_counts.items()}

    #job = execute(circ_list, backend=backend, initial_layout=initial_layout)
    #job = backend.jobs()[0]
    # mit.cals_from_system(initial_layout)

    #result = job.result()
    #pop_counts, mqc_counts = test.get_counts(result)
    #mit.apply_correction(counts, initial_layout)

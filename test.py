# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:27:54 2021

@author: Fidel
"""

# Import standard Qiskit libraries
from qiskit import IBMQ

# Import other standard libraries

# Import local modules
from graphstate import GraphState
import utilities as util

util.startup()
provider = IBMQ.get_provider('ibm-q-melbourne')

device = 'ibmq_manhattan'
backend = provider.get_backend(device)
test = GraphState(backend)
print(f'Device: {device}')
print(f'Number of qubits: {test.nqubits}')
print(f'Number of edges: {test.nedges}')
print(f'List of edges: {test.edges}')
print(test.graphstate_circuit)
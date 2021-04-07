# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:37:39 2021

@author: Fidel
"""

# Standard libraries


# Qiskit libraries
from qiskit import IBMQ

token = '509707245c44e538cd6e320690c2caae9aec5b915172ae11ebbd74591772f59dc7e62e17c94d284d3d802e403476e4e0da9a422fca2d25c5f11ebc2f3e719da4'

def startup(token=token, hub='ibm-q-melbourne', group=None, project=None):
    """Start up session"""
    if IBMQ.active_account() == None:
        IBMQ.enable_account(token)
        print(f'Provider:', hub)
        provider = IBMQ.get_provider(hub)
        check_provider(hub)
        
def check_provider(hub):
    """Check list of providers with queue size and qubit count for input hub"""
    provider = IBMQ.get_provider(hub)
    
    for backend in provider.backends():
      try:
        qubit_count = len(backend.properties().qubits)
      except:
        qubit_count = 'simulated'
    
      print(f'{backend.name()} has {backend.status().pending_jobs} queud and {qubit_count} qubits')
      

    
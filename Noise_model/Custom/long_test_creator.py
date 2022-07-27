import math
import random
from qiskit import (
	IBMQ,
	QuantumCircuit,
	transpile,
)
from qiskit.providers import Backend
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
import numpy as np
from matplotlib.pyplot import figure
import datetime
import os
import time
import pickle
import logging
import sys
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate, ZGate, Reset, Measure
from custom_noise_model_creator import CustomNoiseModelCreator


logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(filename='long_testing_creator.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

def add_random_gate(qc: QuantumCircuit):
	param = random.uniform(0,2) * math.pi
	gates = [XGate(), CXGate()]
	gate = random.choice(gates)
	n_qubits = qc.num_qubits
	if isinstance(gate, CXGate):
		target = random.sample(range(n_qubits), k=2)
		qc.append(gate, target)
	else:
		target = random.choice(list(range(n_qubits)))
		qc.append(gate, [target])

	return qc

def get_circuit_with_length(hw, length=2):
	samples = np.linspace(np.ceil(length/2), length*3, length*5, dtype=int)
	n_qubits = hw.configuration().n_qubits
	for _ in range(100):
		circuits = []
		for sample in samples:
			qc = QuantumCircuit(n_qubits, n_qubits)
			qc_tmp = QuantumCircuit(n_qubits, n_qubits)
			for j in range(sample):
				qc_tmp = add_random_gate(qc_tmp)
			qc.barrier()
			qc.append(qc_tmp, range(n_qubits), range(n_qubits))
			qc.barrier(range(n_qubits))
			qc.append(qc_tmp.inverse(), range(n_qubits), range(n_qubits))
			qc.barrier()
			qc.measure_all(add_bits=False)
			circuits.append(qc)
		c_t = transpile(circuits, hw, optimization_level=0)
		depths = np.array([c.depth() - 1 for c in c_t])
		if length in depths:
			idx = np.nonzero(depths == length)[0]
			choosen_idx = random.choice(idx)
			assert c_t[choosen_idx].depth() - 1 == length
			return c_t[choosen_idx]
		else:
			pass
		
	raise Exception("no circuit found")

def create_test_circuits(hw, start, step, end, samples_per_step):
	samples = np.arange(start, end, step, dtype=int)
	circs_samples = []
	for sample in samples:
		for _ in range(samples_per_step):
			qc = get_circuit_with_length(hw, sample)
			circs_samples.append(qc)
	return circs_samples

class Measurement:
	def __init__(self, circs):
		self.backend_name = None
		self.circuits = circs
		self.noise_model = None
		self.job = None
		self.creator = None
		self.ts = None

	def run(self, backend):
		self.backend_name = backend.name()
		self.ts = datetime.datetime.now()
		self.creator = CustomNoiseModelCreator(backend)
		self.creator.create_test_circuits()
		self.creator.run_circs()
		self.job = backend.run(self.circuits, shots=10000)

		self.noise_model = NoiseModel.from_backend(backend)

provider = IBMQ.load_account()	

backends = provider.backends(n_qubits=5, operational=True, simulator=False)

test_circs = {}
for hw in backends:
	if os.path.isfile(f"test_circs_cnot_{hw.name()}.pickle"):
		logger.debug("loading existing test circs")
		with open(f"test_circs_cnot_{hw.name()}.pickle", "rb") as file:
			test_circs[hw.name()] = pickle.load(file)
	else:
		logger.debug(f"create new test circs for {hw.name()}")
		test_circs[hw.name()] = create_test_circuits(hw, 6, 2, 25, 10)
		with open(f"test_circs_cnot_{hw.name()}.pickle", "wb") as file:
			pickle.dump(test_circs[hw.name()], file)

out_folder = "output_long_experiment_creator"
if not os.path.exists(out_folder):
	os.mkdir(out_folder)

steps = 1000
start = datetime.datetime.now()
try:
	prev_measures = {hw.name(): None for hw in backends}
	for i in range(steps):
		logger.info(f"starting measurement nr {i}")
		all_measures = {}
		for hw in backends:
			if prev_measures[hw.name()] is not None and not prev_measures[hw.name()].job.done():
				#if the test circuits job is done the creator job is done as well
				logger.debug(f"backend {hw.name()} skipped because previous job is not done yet")
				continue 
			logger.info(f"starting measurement for backend {hw.name()}")
			measure = Measurement(test_circs[hw.name()])
			measure.run(hw)
			all_measures[hw.name()] = measure

		with open(os.path.join(out_folder, f"measurement_{i}"), "wb") as file:
			pickle.dump(all_measures, file)

		for k, v in all_measures.items():
			prev_measures[k] = v

		time.sleep(20*60)

except KeyboardInterrupt as e:
	logger.warning(f"stopped after {i} iterations")
	logger.warning(f"started {str(start)}")
	logger.warning(f"end {str(datetime.datetime.now())}")

except Exception as e:
	logger.exception(f"\nException: {e}")

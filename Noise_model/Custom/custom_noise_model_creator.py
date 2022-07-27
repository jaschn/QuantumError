from collections import OrderedDict, defaultdict
import copy
import itertools
from typing import Any
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer.noise import NoiseModel, ReadoutError, QuantumError
from qiskit.providers import Backend
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate, ZGate, Reset, Measure
from qiskit.circuit import Gate
from helpers import get_error_rate, fit_reverse_gates, get_error_rate_per_qubit
import math
import numpy as np
from numpy.polynomial import Polynomial
import logging

class CustomNoiseModelCreator:
	def __init__(self, hw: Backend) -> None:
		self.hw: Backend = hw
		self.n_qubits: int = self.hw.configuration().n_qubits
		self.test_circuits: dict[Any, list[QuantumCircuit]] = OrderedDict()
		self.test_circuits_cnot: dict[Any, list[QuantumCircuit]] = OrderedDict()
		self.results: dict[Any, dict[str, int]] = {}
		self.results_cnot: dict[Any, dict[str, int]] = {}
		self.basis_gates = ["x", "cx", "measure"]
		self.samples = np.array([1, 2, 3, 4, 5, 8, 13, 20])
		self.samples_cnot = np.array([1, 2, 3, 6, 12, 20])
		self.job = None
		self.reverse_fits = [None] + [fit_reverse_gates(i) for i in range(1, 21)]
		self.logger = logging.getLogger("main")

	def _get_independent_qubits(self):
		#probably not the most efficient way
		coupling = self.hw.configuration().coupling_map

		neigbours = defaultdict(list)
		for i in range(self.n_qubits):
			for j in range(self.n_qubits):
				if i == j:
					continue
				if [i,j] in coupling or [j,i] in coupling:
					neigbours[i].append(j)

		visited = []
		independent = []
		for i in range(self.n_qubits):
			if i in visited:
				continue
			non_neigbours = [i]
			visited.append(i)
			for j in range(self.n_qubits):
				if j == i or j in visited:
					continue
				valid = True
				for x in non_neigbours:
					if j in neigbours[x]:
						valid = False
				if valid:
					non_neigbours.append(j)
					visited.append(j)
			independent.append(non_neigbours)

		return independent

	def _create_readout_test_circuits(self):
		independent = self._get_independent_qubits()

		for i in independent:
			prep1_measure_circs = []
			for j in self.samples:
				qc = QuantumCircuit(self.n_qubits, self.n_qubits)
				for _ in range(j):
					qc.x(i)
					qc.barrier()
				qc.measure(i, i)
				prep1_measure_circs.append(qc)
			self.test_circuits[tuple(i)] = prep1_measure_circs

		prep0_measure = []
		for i in range(self.n_qubits):
			qc = QuantumCircuit(self.n_qubits, 1)
			qc.measure(i, 0)
			prep0_measure.append(qc)
		self.test_circuits["prep0_measure"] = prep0_measure


	def _create_cnot_test_circs(self):
		coupling_map = self.hw.configuration().coupling_map
		independent_coupling = []
		visited_couplings = []
		for c in coupling_map:
			if c in visited_couplings:
				continue
			i = [c]
			visited_couplings.append(c)
			visited_couplings.append([c[1], c[0]])
			for cc in coupling_map:
				if cc in visited_couplings:
					continue
				valid = True
				for ccc in i:
					if cc[0] in ccc or cc[1] in ccc:
						valid = False
						break
				if valid:
					i.append(cc)
					visited_couplings.append(cc)
					visited_couplings.append([cc[1], cc[0]])

			independent_coupling.append(i)

		for cs in independent_coupling:
			circs = []
			for i in self.samples_cnot:
				qc = QuantumCircuit(self.n_qubits, self.n_qubits)
				for _ in range(i):
					for c in cs:
						qc.cx(c[0], c[1])
					qc.barrier()
				for c in cs:
					qc.measure([c[0], c[1]], [c[0], c[1]])
				circs.append(qc)

			self.test_circuits_cnot[tuple([tuple(t) for t in cs])] = circs


	def create_test_circuits(self):
		self._create_readout_test_circuits()
		self._create_cnot_test_circs()

	def _combine_circs(self):
		combined = []
		for v in self.test_circuits.values():
			combined.extend(v)

		for v in self.test_circuits_cnot.values():
			combined.extend(v)

		return combined

	def run_circs(self):
		combined = self._combine_circs()
		assert len(combined) <= 100, "Splitting into multiple jobs not supported"
		self.logger.debug(f"number of test circuits {len(combined)}")
		combined_transpiled = transpile(combined, self.hw, optimization_level=0)
		self.job = self.hw.run(combined_transpiled, shots=10000)

	def _calc_error_rates(self):
		self.not_errors = [None] * self.n_qubits

		for k, v in self.results.items():
			if not isinstance(k, tuple):
				continue
			qubit_results = []
			for j, r in enumerate(v):
				if self.samples[j] % 2 == 0:
					qubit_results.append(get_error_rate_per_qubit(r, {"0"*self.n_qubits : 1}))
				else:
					qubit_results.append(get_error_rate_per_qubit(r, {"1"*self.n_qubits : 1}))
			
			for i in k:
				assert self.not_errors[i] == None, "qubit was tested in two different circuits"
				self.not_errors[i] = np.array(qubit_results)[:, i]

		self.not_errors = np.array(self.not_errors)
		self.odd_idx = self.samples % 2 == 1
		self.even_idx = self.samples % 2 == 0
		self.odd_errors = self.not_errors[:, self.odd_idx]
		self.even_errors = self.not_errors[:, self.even_idx]

		odd_fits = [Polynomial.fit(self.samples[self.odd_idx], self.odd_errors[i], 2) for i in range(self.n_qubits)]
		self.prep1_meas_error = np.array([odd_fits[i](0) for i in range(self.n_qubits)])
		self.prep0_meas_error = np.array([get_error_rate(r, {"0": 1}) for r in self.results["prep0_measure"]])

		self.error_cnot = {}
		for k, v in self.results_cnot.items():
			e = np.array([get_error_rate_per_qubit(r, {"0"*self.n_qubits: 1}) for r in v])
			for c in k:
				self.error_cnot[c] = e[:, c]

	def _fix_readout_error(self, error: np.ndarray, expected=0, qubit=None, clip_values=False):
		if expected != 0 and expected != 1:
			raise Exception(f"The expected reaoud is not valid {expected}")

		if qubit != None:
			if expected == 0:
				real = (error - (1 - self.prep1_meas_error)[qubit]) / (self.prep0_meas_error - (1 - self.prep1_meas_error))[qubit]
			else:
				raise NotImplementedError()
		else:
			if expected == 0:
				real = (error - (1 - self.prep1_meas_error)[:, np.newaxis]) / (self.prep0_meas_error - (1 - self.prep1_meas_error))[:, np.newaxis]
			else: 
				real = (error - (1 - self.prep0_meas_error)[:, np.newaxis]) / (self.prep1_meas_error - (1 - self.prep0_meas_error))[:, np.newaxis]

		if clip_values:
			real[real>1] = 1
			real[real<0] = 0

		assert real.min() >= 0, "_fix_readout_error under 0"
		assert real.max() <= 1, "_fix_readout_error over 1"
		return real

	def _get_readout_errors(self):
		readout_errors = []

		for i in range(self.n_qubits):
			probabilities = [[self.prep0_meas_error[i], 1-self.prep0_meas_error[i]], [1-self.prep1_meas_error[i], self.prep1_meas_error[i]]]
			readout_errors.append(([i], ReadoutError(probabilities)))

		return readout_errors

	def _get_xgate_error_rate(self):
		fixed_error_even = self._fix_readout_error(self.even_errors, expected=0, clip_values=True)
		fixed_error_odd = self._fix_readout_error(self.odd_errors, expected=1, clip_values=True)

		combined = np.empty_like(self.not_errors)
		combined[:, self.even_idx] = fixed_error_even
		combined[:, self.odd_idx] = fixed_error_odd

		selected_idx = self.samples<len(self.reverse_fits)

		selected_l = self.samples[selected_idx]
		pred = np.zeros(self.n_qubits)
		for i in range(self.n_qubits):
			selected = combined[i, selected_idx]
			selected = selected[selected>0.8]
			pred[i] = np.array([self.reverse_fits[i](s) for i, s in zip(selected_l, selected)]).mean()

		x_error = 1 - pred
		x_error[x_error<0] = 0
		x_error[x_error>1] = 1

		return x_error

	def _get_gate_error(self, get_errors):
		x_error = get_errors()
		no_error = 1 - x_error

		q_errors = []
		for i in range(self.n_qubits):
			noise_ops = [(IGate(), no_error[i]),
						 (XGate(), x_error[i])]
			q_error = QuantumError(noise_ops, number_of_qubits=1)
			q_errors.append(q_error)

		return q_errors

	def _get_cxgate_error(self):

		err_f = {}
		for k, e in self.error_cnot.items():
			err_f[k] = self._fix_readout_error(e[:, 0], 0, 0, clip_values=True) * self._fix_readout_error(e[:, 1], 0, 1, clip_values=True)

		error_inv_fit = {}
		for k, e in err_f.items():
			selected_idx = np.logical_and(self.samples_cnot<len(self.reverse_fits), e>0.8)
			selected_l = self.samples_cnot[selected_idx]
			selected = e[selected_idx]
			error_inv_fit[k] =  np.array([self.reverse_fits[f](err) for f, err in zip(selected_l, selected)]).mean()

		error = {}

		for k, v in error_inv_fit.items():
			error[k] = [v, (1 - np.sqrt(v))**2, (1 - np.sqrt(v))*np.sqrt(v), (1 - np.sqrt(v))*np.sqrt(v)]
		
		qc_no_error = QuantumCircuit(2)
		qc_no_error.id([0,1])

		qc_error_first = QuantumCircuit(2)
		qc_error_first.x(0)
		qc_error_first.id(1)

		qc_error_second = QuantumCircuit(2)
		qc_error_second.id(0)
		qc_error_second.x(1)

		qc_error_both = QuantumCircuit(2)
		qc_error_both.x(0)
		qc_error_both.x(1)

		q_errors = {}
		for k, v in error.items():
			noise_ops = [(qc_no_error, error[k][0]),
						 (qc_error_both, error[k][1]),
						 (qc_error_first, error[k][2]),
						 (qc_error_second, error[k][3])]
			q_error = QuantumError(noise_ops)
			q_errors[k] = q_error

		return q_errors
			 
	def get_custom_noise_model(self, noise_decoherence: NoiseModel = None):
		results = self.job.result().get_counts()
		cnt = 0
		for k, v in self.test_circuits.items():
			l = len(v)
			self.results[k] = results[cnt:cnt+l]
			cnt += l

		for k, v in self.test_circuits_cnot.items():
			l = len(v)
			self.results_cnot[k] = results[cnt:cnt+l]
			cnt += l

		self._calc_error_rates()

		noise = NoiseModel(basis_gates=self.basis_gates)

		for r in self._get_readout_errors():
			noise.add_readout_error(r[1], r[0])

		gate_errors = self._get_gate_error(self._get_xgate_error_rate)
		for i, e in enumerate(gate_errors):
			noise.add_quantum_error(e, "x", [i])

		gate_errors = self._get_cxgate_error()
		for i, e in gate_errors.items():
			noise.add_quantum_error(e, "cx", i)
			noise.add_quantum_error(e, "cx", (i[1], i[0]))


		return noise


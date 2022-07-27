import matplotlib.pyplot as plt
import numpy as np
from helpers import get_error_rate
from matplotlib.pyplot import figure
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import copy
from modify_probs_functions import modify_relaxation, number_of_noise_params, modify_noise_model_local, modify_noise_model_readout
import pickle
import logging

class NoiseModelImprove:
	def __init__(self, start_noise_model: NoiseModel, circs: list[QuantumCircuit], error_rate_hw: list[float], tmp_folder="tmp") -> None:
		self.logger = logging.getLogger("main")
		self.current_best_noise = copy.deepcopy(start_noise_model)
		self.circs = circs
		self.expected_error_hw = np.array(error_rate_hw)
		assert len(self.circs) == len(self.expected_error_hw)
		self.current_best_loss = self._loss(self.current_best_noise)
		self.loss_hist = [self.current_best_loss]
		self.local_cnt, self.readout_cnt = number_of_noise_params(self.current_best_noise)
		self.n_qubits = len(self.current_best_noise.noise_qubits)
		self.counter = 0
		self.tmp_folder = tmp_folder

	def _loss(self, noise_model: NoiseModel) -> float:
		self.logger.debug("create sim based on trial noise model")
		sim = AerSimulator(noise_model=noise_model, seed_simulator=42)
		self.logger.debug("run sim for loss")
		results = sim.run(self.circs).result().get_counts()
		self.logger.debug("sim for loss done")
		sim_error = np.array([get_error_rate(r, {"00000" : 1}) for r in results])
		diff = np.square(sim_error-self.expected_error_hw).mean()
		return diff

	def _check_if_better(self, new_noise_model: NoiseModel) -> bool:
		new_loss = self._loss(new_noise_model)
		if new_loss < self.current_best_loss:
			self.logger.debug(f"\nfound better model with loss: {new_loss}")
			self.current_best_noise = new_noise_model
			self.current_best_loss = new_loss
			self.loss_hist.append(self.current_best_loss)
			#self.counter += 1
			# if self.counter >= 10:
			# 	with open(f"./{self.tmp_folder}/current_best_noise_model_after{len(self.loss_hist)}.pickle", "wb") as file:
			# 		pickle.dump(self.current_best_noise, file)
			# 	self.counter = 0
			return True
		else:
			self.loss_hist.append(self.current_best_loss)
			return False

	def save_loss_hist_graph(self, file):
		figure(figsize=(10, 10), dpi=100)
		plt.title("Loss history with structured search")
		plt.ylabel("Loss")
		plt.xlabel("Number of search steps")
		plt.plot(self.loss_hist, label="Loss history")
		plt.ylim(bottom=0, top=max(self.loss_hist)*1.1)
		plt.legend()
		plt.savefig(file)

	def structured_search(self, start_factor, end_factor):
		factor = start_factor
		self.logger.debug(f"starting with loss: {self.current_best_loss}")
		while factor >= end_factor:
			self.logger.debug(f"new factor {factor}")
			for i in range(self.readout_cnt):
				for j in range(2):
					for _ in range(10):
						self.logger.debug(".")
						new_noise_model = modify_noise_model_readout(copy.deepcopy(self.current_best_noise), i, j, factor)
						if self._check_if_better(new_noise_model):
							continue
						else:
							break
			self.logger.debug("done with readout postive factor")

			for i in range(self.readout_cnt):
				for j in range(2):
					for _ in range(10):
						self.logger.debug(".")
						new_noise_model = modify_noise_model_readout(copy.deepcopy(self.current_best_noise), i, j, -factor)
						self.logger.debug("chech for better")
						if self._check_if_better(new_noise_model):
							self.logger.debug("found better")
							continue
						else:
							self.logger.debug("no better found")
							break
			self.logger.debug("done with readout negative factor")
			
			for i, cnt in enumerate(self.local_cnt):
				if cnt == 1: 
					continue
				for j in range(10):
					self.logger.debug(".")
					new_noise_model = modify_noise_model_local(copy.deepcopy(self.current_best_noise), i, factor)
					if self._check_if_better(new_noise_model):
						continue
					else:
						break
			
			self.logger.debug("done with local postive factor")

			for i, cnt in enumerate(self.local_cnt):
				if cnt == 1: 
					continue
				for j in range(10):
					self.logger.debug(".")
					new_noise_model = modify_noise_model_local(copy.deepcopy(self.current_best_noise), i, -factor)
					if self._check_if_better(new_noise_model):
						continue
					else:
						break

			self.logger.debug("done with local negative factor")
			factor *= 0.1
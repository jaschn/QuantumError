import logging
import numpy as np
from qiskit.providers.aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit.providers.aer.noise.passes.relaxation_noise_pass import RelaxationNoisePass

logger = logging.getLogger("main")

def number_of_noise_params(noise_model: NoiseModel):
	local_params = []
	for name, qubit_dict in noise_model._local_quantum_errors.items():
		for qubits, error in qubit_dict.items():
			local_params.append(len(error.probabilities))

	readout_params = len(noise_model._local_readout_errors)
	return local_params, readout_params

def modify_quantum_error_probs(error: QuantumError, factor):
	if len(error.probabilities) == 1:
		raise Exception("error probs len 1")
	probs = np.array(error.probabilities)
	probs_new = np.zeros(len(probs))
	missing = probs[0]*factor
	probs_new[0] = probs[0]*(1+factor)
	if probs_new[0] > 1:
		probs_new[0] = 1
		probs_new[0+1:] = 0
	else:
		sum_except_main = probs[1:].sum()
		if sum_except_main != 0:
			probs_new[1:] = probs[1:] - missing*(probs[1:]/sum_except_main)
		else:
			probs_new[1:] = probs[1:] - missing*(1/(len(probs)-1))

	assert probs_new.min() >= 0, "modify_quantum_error_probs: probs_new.min() >= 0"
	assert probs_new.max() <= 1, "modify_quantum_error_probs: probs_new.max() <= 1"
	#probs_new = probs_new / probs_new.sum()
	assert np.isclose(probs_new.sum() - 1, 0, atol=0.00000001), f"props total {probs_new.sum()}"
	error._probs = probs_new
	return error

def modify_noise_model_local(noise_model, index, factor):
	assert len(noise_model._default_quantum_errors.keys()) == 0, "modify_noise_model_local: len(noise_model._default_quantum_errors.keys()) == 0"
	assert len(noise_model._nonlocal_quantum_errors.keys()) == 0, "modify_noise_model_local: len(noise_model._default_quantum_errors.keys()) == 0"
	assert noise_model._default_readout_error is None, "modify_noise_model_local: len(noise_model._default_quantum_errors.keys()) == 0"

	cnt = 0
	for name, qubit_dict in noise_model._local_quantum_errors.items():
		if index >= cnt + len(qubit_dict):
			cnt += len(qubit_dict)
			continue
		for qubits, error in qubit_dict.items():
			if cnt == index:
				qubit_dict[qubits] = modify_quantum_error_probs(error, factor)
				cnt += 1
				break
			else:
				cnt += 1
		noise_model._local_quantum_errors[name] = qubit_dict
		break
	
	assert cnt-1 == index, "modify_noise_model_local: cnt-1 == index"
	return noise_model

def modify_readout_error_probs(error: ReadoutError, change_expected, factor):
	expect_0, expect_1 = error.probabilities
	if change_expected != 0 and change_expected != 1:
		raise Exception()

	if change_expected == 0:
		expect_0_new = [expect_0[0] * (1+factor), expect_0[1]-expect_0[0]*factor]
		if expect_0_new[0] > 1:
			expect_0_new = [1,0]
		expect_1_new = expect_1
	else:
		expect_1_new = [expect_1[0]-expect_1[1]*factor, expect_1[1] * (1+factor)]
		if expect_1_new[1] > 1:
			expect_1_new = [0,1]
		expect_0_new = expect_0
	probs_new = np.array([expect_0_new, expect_1_new])

	assert probs_new.min() >= 0, "modify_readout_error_probs: probs_new.min() >= 0"
	assert probs_new.max() <= 1, "modify_readout_error_probs: probs_new.max() <= 1"
	#probs_new[0] = probs_new[0] / probs_new[0].sum()
	#probs_new[1] = probs_new[1] / probs_new[1].sum()
	assert np.isclose(probs_new.sum() - 2, 0, atol=0.00000001), f"props total {probs_new.sum()}"
	logger.debug(f"change_expected: {change_expected}")
	logger.debug(f"factor: {factor}")
	logger.debug(f"error.probabilities: {error.probabilities}")
	logger.debug(f"probs_new: {probs_new}")
	error._probabilities = probs_new
	return error

def modify_noise_model_readout(noise_model, index, change_expected, factor):
	assert len(noise_model._default_quantum_errors.keys()) == 0, "modify_noise_model_readout: len(noise_model._default_quantum_errors.keys()) == 0"
	assert len(noise_model._nonlocal_quantum_errors.keys()) == 0, "modify_noise_model_readout: len(noise_model._nonlocal_quantum_errors.keys()) == 0"
	assert noise_model._default_readout_error is None, "modify_noise_model_readout: noise_model._default_readout_error is None"
	assert index < len(noise_model._local_readout_errors), "modify_noise_model_readout: index < len(noise_model._local_readout_errors)"
	
	cnt = 0
	for qubits, error in noise_model._local_readout_errors.items():
		if cnt == index:
			noise_model._local_readout_errors[qubits] = modify_readout_error_probs(error, change_expected, factor)
			break
		else:
			cnt += 1

	assert cnt == index, "modify_noise_model_readout: cnt == index"
	return noise_model

def modify_relaxation(noise: NoiseModel, param, index, factor):
	assert len(noise._custom_noise_passes) == 1, "modify_relaxation: len(noise._custom_noise_passes) == 1"
	relax = noise._custom_noise_passes[0]
	if param == "p1s":
		values = relax._p1s
	elif param == "t1s":
		values = relax._t1s
	elif param == "t2s":
		values = relax._t2s
	else:
		raise Exception()

	new_values = values.copy()
	new_values[index] = values[index]*(1+factor)

	if param == "p1s":
		relax._p1s = new_values
	elif param == "t1s":
		relax._t1s = new_values
	else:
		relax._t2s = new_values

	noise._custom_noise_passes[0] = relax

	return noise
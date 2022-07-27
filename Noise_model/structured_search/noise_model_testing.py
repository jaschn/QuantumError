import pickle
import numpy as np
from helpers import get_error_rate
import itertools
from NoiseModelImprove import NoiseModelImprove

with open("more_detailed_error_all.pickle", "rb") as file:
	circs_samples_all, jobs_samples_all, circs_result_all, noise_model = pickle.load(file)

circs = list(itertools.chain.from_iterable(circs_samples_all.values()))
hw_result = list(itertools.chain.from_iterable(circs_result_all.values()))
hw_error = np.array([get_error_rate(r, {"00000" : 1}) for r in hw_result])

improv = NoiseModelImprove(noise_model, circs, hw_error)
try:
	print("start search")
	improv.structured_search(0.1, 0.000001)
	print("end search")
	improv.save_loss_hist_graph("./loss_hist.png")
	print("saved graph")
except KeyboardInterrupt:
	print("exit by keyboard interrupt")
	exit()
except Exception as e:
	print("exception")
	print(e)

with open("updated_improved_noise_model.pickle", "wb") as file:
	pickle.dump(improv.current_best_noise, file)
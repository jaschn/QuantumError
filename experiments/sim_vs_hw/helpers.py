def get_error_rate(result: dict[str, int], perfect_result: dict[str, int]) -> float:
	sum_results = sum(result.values())
	sum_perfect = sum(perfect_result.values())
	result = {k: v/sum_results for k, v in result.items()}
	perfect_result = {k: v/sum_perfect for k, v in perfect_result.items()}
	diffs = []
	values = set(list(result.keys()) + list(perfect_result.keys()))
	for k in values:
		if k in result:
			if k in perfect_result:
				diff = abs(result[k] - perfect_result[k])
			else:
				diff = result[k]
		else:
			diff = perfect_result[k]
		diffs.append(diff)
	error_rate = 1 - sum(diffs)/2
	return error_rate
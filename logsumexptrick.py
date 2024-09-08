# How to add a list of probabilities together that are 
# represented as "log probabilities", to avoid underflow.

from sys import float_info
from math import log, exp

# This effectively acts as probability 0 in the form of log probability.
min_log_prob = -float_info.max

# Adding a list of probabilities represented as log probabilities.
def logsumexp(vals):
	if len(vals) == 0:
		return min_log_prob
	m = max(vals)
	if m == min_log_prob:
		return min_log_prob
	else:
		return m + log(sum([exp(val - m) for val in vals]))


# Example of use.
if __name__ == '__main__':
	# Two probabilities.
	prob1 = 0.0001
	prob2 = 0.000001

	# Now represented as log probabilities.
	logprob1 = log(prob1)
	logprob2 = log(prob2)

	# The log of the sum of the two probabilities.
	logsummed = logsumexp([logprob1, logprob2])
	 
	# Converting log probability back to probability by exponentiation.
	summed = exp(logsummed)
	print("prob1 + prob2 =", summed)

	# Multiplication of probabilities is simply by adding
	# log probabilities together.
	logmultiplied = logprob1 + logprob2
	multiplied = exp(logmultiplied)
	print("prob1 * prob2 =", multiplied)

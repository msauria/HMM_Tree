import sys

import numpy


class InitialProbability():

    def __init__(self, num_states=None, initial_probabilities=None, initial_mapping=None, seed=None):
        assert (num_states is not None) != (initial_probabilities is not None)
        if num_states is not None:
            RNG = numpy.random.default_rng(seed)
            self.initial_probabilities = RNG.rand(size=num_states).astype(numpy.float64)
            self.initial_probabilities /= numpy.sum(self.initial_probabilities)
            self.initial_mapping = numpy.arange(num_states)
        else:
            self.initial_probabilities = initial_probabilities.astype(numpy.float64)
            if initial_mapping is not None:
                self.initial_mapping = initial_mapping.astype(numpy.int32)
                self.initial_probabilities[numpy.where(self.initial_mapping < 0)[0]] = 0
        self.initial_probabilities /= numpy.sum(self.initial_probabilities)
        self.log_initprobs = numpy.full(num_states, -numpy.inf, numpy.float64)
        where = numpy.where(self.initial_probabilities > 0)[0]
        self.log_initprobs[where] = numpy.log(self.initial_probabilities[where])
        self.updated = False
        self.tallies = numpy.zeros(self.initial_probabilities.shape[0], numpy.float64)
        return

    def __getitem__(self, idx):
        return self.initial_logprobs[idx]

    def __setitem__(self, idx, value):
        self.initial_probabilities[idx] = value
        return

    def clear_tallies(self):
        self.updated = False
        self.tallies.fill(0)
        return

    def update_tallies(self, probs):
        self.tallies[:] = numpy.sum(probs, axis=0)
        return

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        where = numpy.where(self.initial_mapping >= 0)[0]
        tmp = numpy.bincount(self.initial_mapping[where],
                             weights=self.tallies)
        tmp /= numpy.maximum(numpy.bincount(self.initial_mapping[where]), 1)
        self.initial_probabilities.fill(0)
        self.initial_probabilities[where] = tmp[self.initial_mapping[where]]
        self.initial_probabilities /= numpy.sum(self.initial_probabilities)
        where = numpy.where(self.initial_probabilities > 0)[0]
        self.initial_logprobs.fill(-numpy.inf)
        self.initial_logprobs[where] = numpy.log(self.initial_probabilities[where])
        return
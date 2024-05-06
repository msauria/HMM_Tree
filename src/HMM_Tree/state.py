import numpy

class State():
    """Base class for hmm state"""

    def __init__(self, distributions, index, label=""):
        if type(distributions) == list:
            self.distributions = distributions
        else:
            self.distributions = [distributions]
        self.num_distributions = len(self.distributions)
        self.label = label
        self.index = int(index)
        return

    def score_observations(self, obs, *args):
        start, end, probs_shape, mixed_distN, smm_map = args
        names = obs.dtype.names
        probs = self.distributions[0].score_observations(obs[names[0]], *args)
        for i in range(1, self.num_distributions):
            probs += self.distributions[i].score_observations(obs[names[i]], *args)
        return probs

    def clear_tallies(self):
        for i in range(self.num_distributions):
            self.distributions[i].clear_tallies()
        return

    def update_tallies(self, obs, total):
        names = obs.dtype.names
        for i in range(self.num_distributions):
            if not self.distributions[i].fixed:
                self.distributions[i].update_tallies(obs[names[i]], total)
        return

    def apply_tallies(self):
        for i in range(self.num_distributions):
            if not self.distributions[i].fixed:
                self.distributions[i].apply_tallies()
        return

    def print(self, level=0):
        output = []
        output.append(f"{' '*level}State {self.index} {self.label}")
        for i in range(len(self.distributions)):
            output.append(f"{' '*(level + 1)}Distribution {i}")
            output.append(self.distributions[i].print(level + 2))
        output = "\n".join(output)
        return output

    def generate_sequence(self, RNG=numpy.random):
        obs = []
        for i in range(self.num_distributions):
            obs.append(self.distributions[i].generate_emission(RNG))
        return tuple(obs)

    def get_parameters(self):
        params = []
        for i in range(self.num_distributions):
            params.append(self.distributions.get_parameters())
        return params
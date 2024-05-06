import sys
from multiprocessing.shared_memory import SharedMemory

import numpy

from .emission import *
from .state import State
from .transition import TransitionMatrix

class Node():
    """Base class for tree hmm model node"""

    def __init__(self, label, index=0):
        self.name = 'Node'
        self.label = label
        self.parent = None
        self.parent_name = None
        self.children = []
        self.children_names = []
        self.index = index
        self.level = None
        self.root = False
        self.leaf = False
        self.features = None
        self.views = {}
        return

    def find_levels(self, level=0):
        if self.level is None:
            self.level = level
        else:
            raise RuntimeError(f"Node {self.name} appears in tree in multiple positions")
        if level == 0:
            self.root = True
        level += 1
        for n in self.children:
            n.find_levels(level)
        if len(self.children) == 0:
            self.leaf = True
        return

    def print_tree(self):
        if self.level is None:
            level = 0
        else:
            level = self.level
        output = f"{level * '  '}{self.label}\n"
        for c in self.children:
            output += c.print_tree()
        return output

    def find_pairs(self):
        pairs = {}
        if self.parent is not None:
            pairs[self.name] = self.parent.name
        for c in self.children:
            pairs.update(c.find_pairs())
        return pairs

    def initialize_HMM(self,
                       state_names=None,
                       emissions=None,
                       transition_matrix=None,
                       initial_probabilities=None,
                       seed=None):
        self.RNG = numpy.random.default_rng(seed)
        self.num_states = len(emissions)
        self.num_dists = 0
        self.states = []
        for h, E in enumerate(emissions):
            assert (issubclass(type(E), EmissionDistribution) or
                    type(E) == list)
            if issubclass(type(E), EmissionDistribution):
                E = [E]
            for E0 in E:
                assert issubclass(type(E0), EmissionDistribution)
                if E0.index is not None:
                    continue
                E0.index = self.num_dists
                self.num_dists += 1
            self.num_emissions = len(E)
            self.states.append(State(E, h))
            if state_names is not None and h < len(state_names):
                self.states[-1].label = state_names[h]
            else:
                raise RuntimeError("emissions must be None, an Emission instance, or a list of Emission instances")
        self.distributions = [None for x in range(self.num_dists)]
        for i in range(self.num_states):
            for D in self.states[i].distributions:
                if self.distributions[D.index] is None:
                    self.distributions[D.index] = D
        self.initial_probabilities = initial_probabilities
        self.transitions = transition_matrix
        self.tallies = numpy.zeros(self.num_states, numpy.float64)
        return

    def find_levels(self, level=0):
        if self.level is None:
            self.level = level
        else:
            raise RuntimeError(f"Node {self.name} appears in tree in multiple positions")
        if level == 0:
            self.root = True
        level += 1
        for n in self.children:
            n.find_levels(level)
        if len(self.children) == 0:
            self.leaf = True
        return

    def get_emission_dtype(self):
        dtypes = []
        for i in range(self.states[0].num_distributions):
            if issubclass(type(self.states[0].distributions[i]), EmissionContinuousDistribution):
                dtypes.append(numpy.float64)
            else:
                dtypes.append(numpy.int32)
        return numpy.dtype([(f"{x}", dtypes[x]) for x in range(len(dtypes))])

    @classmethod
    def generate_sequence(cls, *args):
        views = []
        start, end, obsN, hmm, smm_map, seed = args
        RNG = numpy.random.default_rng(seed)
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((obsN,), hmm.get_emission_dtype(), buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_states']))
        states = numpy.ndarray((obsN,), numpy.int32, buffer=views[-1].buf)
        state = numpy.searchsorted(numpy.cumsum(hmm.initial_probabilities), RNG.random())
        for i in range(start, end):
            states[i] = state
            obs[i] = hmm.states[state].generate_sequence(RNG)
            state = hmm.transitions.generate_transition(state, RNG)
        for V in views:
            V.close()
        return

    def clear_tallies(self):
        for S in self.states:
            S.clear_tallies()
        self.transitions.clear_tallies()
        self.tallies[:] = 0
        return

    def update_tallies(self, *args):
        probs_shape, obs_indices, smm_map = args
        obsN, num_states, num_nodes = probs_shape
        views = []
        views.append(SharedMemory(smm_map['total']))
        total = numpy.ndarray(probs_shape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        self.initial_probabilities.update_tallies(total[obs_indices[:-1], :, self.index])
        return


    def apply_tallies(self):
        for S in self.states:
            S.apply_tallies()
        self.transitions.apply_tallies()
        self.initial_probabilities.apply_tallies()
        return

    def get_parameters(self):
        params = []
        for i in range(self.num_states):
            params.append(self.states[i].get_parameters())
        return params

    @classmethod
    def find_paths(self, *args):
        start, end, index, numObs, obsDtype, probs_shape, node, smm_map = args
        obsN, num_states, num_nodes = probs_shape
        node_idx = node.index
        views = []
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray((obsN,), dtype=obsDtype, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probs_shape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray((probs_shape), dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_states']))
        states = numpy.ndarray((obsN,), dtype=numpy.int32,
                               buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_scores']))
        scores = numpy.ndarray((numObs,), dtype=numpy.float64,
                               buffer=views[-1].buf)
        paths = numpy.zeros((end - start - 1, hmm.num_states), numpy.int32)
        probs[start, :, 0] += hmm.initial_logprobs
        scale = 0
        for i in range(start + 1, end):
            max_val = numpy.amax(probs[i-1, :, 0])
            probs[i-1, :, 0] -= max_val
            scale += max_val
            tmpprobs = probs[i-1:i, :, 0].T + hmm.transitions[:, :] + probs[i:i+1, :, 0]
            best = numpy.argmax(tmpprobs, axis=0)
            paths[i - 1 - start, :] = best
            probs[i, :, 0] = tmpprobs[best, numpy.arange(probs.shape[1])]
        states[end - 1] = numpy.argmax(probs[end - 1, :, 0])
        scores[index] = probs[end - 1, states[end - 1], 0] + scale
        for i in range(start, end - 1)[::-1]:
            states[i] = paths[i - start, states[i + 1]]
        for V in views:
            V.close()
        return

    @classmethod
    def find_probs(self, *args):
        start, end, obsDtype, probs_shape, treeweight_shape, node, smm_map = args
        obsN, num_states, num_nodes = probs_shape
        treeseqN, num_states, num_nodes = treeweight_shape
        node_idx = node.index
        views = []
        views.append(SharedMemory(smm_map['obs']))
        obs = numpy.ndarray(obsN, obsDtype, buffer=views[-1].buf)
        if 'obs_mask' in smm_map:
            views.append(SharedMemory(smm_map['obs_mask']))
            obs_mask = numpy.ndarray((obsN, node.num_states), bool,
                                     buffer=views[-1].buf)
        else:
            obs_mask = None
        marks = obs.dtype.names
        kwargs = {'start': start, "end": end, "smm_map": smm_map}
        views.append(SharedMemory(smm_map['dist_probs']))
        dist_probs = numpy.ndarray((obsN, node.num_dists, num_nodes),
                                   numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray((obsN, node.num_states, num_nodes), numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_weights']))
        tree_weights = numpy.ndarray(treeweight_shape, numpy.float64,
                                     buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_seqs']))
        tree_seqs = numpy.ndarray((treeseqN,), numpy.int32, buffer=views[-1].buf)
        updated = numpy.zeros(node.num_dists, bool)
        state_updated = {}
        for i in range(num_states):
            indices = []
            for j, D in enumerate(node.states[i].distributions):
                indices.append(D.index)
                if updated[D.index]:
                    continue
                if D.updated and D.fixed:
                    continue
                dist_probs[start:end, D.index, node_idx] = D.score_observations(
                    obs[marks[j]][:, node_idx], **kwargs)
                updated[D.index] = True
            indices = tuple(indices)
            if indices in state_updated:
                probs[start:end, i, node_idx] = numpy.copy(
                    probs[start:end, state_updated[indices], node_idx])
            else:
                probs[start:end, i, node_idx] = numpy.sum(
                    dist_probs[start:end, indices, node_idx], axis=1)
                state_updated[indices] = i
        s = numpy.searchsorted(tree_seqs, start)
        e = numpy.searchsorted(tree_seqs, end, side='right')
        probs[tree_seqs[s:e], :, node_idx] += tree_weights[s:e, :, node_idx]
        if obs_mask is not None:
            where = numpy.where(numpy.logical_not(obs_mask[start:end, :]))
            probs[where[0] + start, where[1], :] = -numpy.inf
        for V in views:
            V.close()
        return

    @classmethod
    def find_forward(self, *args):
        start, end, obsDtype, probs_shape, obs_indices, node, smm_map = args
        obsN, num_states, num_nodes = probs_shape
        node_idx = node.index
        views = []
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probs_shape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['forward']))
        forward = numpy.ndarray(probs_shape, dtype=numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((obsN, num_nodes), dtype=numpy.float64,
                              buffer=views[-1].buf)
        valid_trans = node.transitions.valid_trans
        transitions = node.transitions[valid_trans]
        tmp = numpy.full((node.num_states, node.num_states), -numpy.inf, numpy.float64)
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            forward[s, :, node_idx] = node.initial_probabilities[:] + probs[s, :, node_idx]
            scale[s, node_idx] = scipy.special.logsumexp(forward[s, :, node_idx])
            forward[s, :, node_idx] -= scale[s, node_idx]
            for j in range(s + 1, e):
                tmp[valid_trans] = (forward[j - 1, valid_trans[0], node_idx] +
                                    transitions + probs[j, valid_trans[1], node_idx])
                forward[j, :, node_idx] = (scipy.special.logsumexp(tmp, axis=0) +
                                           probs[j, :, node_idx])
                scale[j, node_idx] = scipy.special.logsumexp(forward[j, :, node_idx])
                forward[j, :, node_idx] -= scale[j, node_idx]
        for V in views:
            V.close()
        return

    @classmethod
    def find_reverse(self, *args):
        start, end, obsDtype, probs_shape, obs_indices, node, smm_map = args
        obsN, num_states, num_nodes = probs_shape
        node_idx = node.index
        views = []
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probs_shape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray(probs_shape, dtype=numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((obsN, num_nodes), dtype=numpy.float64,
                              buffer=views[-1].buf)
        valid_trans = node.transitions.valid_trans
        transitions = node.transitions[valid_trans]
        tmp = numpy.full((node.num_states, node.num_states), -numpy.inf, numpy.float64)
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            reverse[e - 1, :, node_idx] = 0
            for j in range(s, e - 1)[::-1]:
                tmp[valid_trans] = ((reverse[j + 1, :, node_idx] +
                                     probs[j + 1, :, node_idx] -
                                     scale[j + 1, node_idx])[valid_trans[1]] +
                                    transitions)
                reverse[j, :, node_idx] = scipy.special.logsumexp(tmp, axis=1)
        for V in views:
            V.close()
        return

    @classmethod
    def find_total(self, *args):
        start, end, obsDtype, probs_shape, obs_indices, node, smm_map = args
        obsN, num_states, num_nodes = probs_shape
        node_idx = node.index
        views = []
        views.append(SharedMemory(smm_map['forward']))
        forward = numpy.ndarray(probs_shape, dtype=numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray(probs_shape, dtype=numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['log_total']))
        log_total = numpy.ndarray(probs_shape, dtype=numpy.float64,
                                  buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['total']))
        total = numpy.ndarray(probs_shape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((obsN, num_nodes), dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_scores']))
        obs_scores = numpy.ndarray((obs_indices.shape[0] - 1, num_nodes, 2),
                                   dtype=numpy.float64, buffer=views[-1].buf)
        valid_trans = node.transitions.valid_trans
        transitions = node.transitions[valid_trans]
        s = obs_indices[start]
        e = obs_indices[end]
        log_total[s:e, :, node_idx] = (forward[s:e, :, node_idx] +
                                       reverse[s:e, :, node_idx])
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            tmp = scipy.special.logsumexp(log_total[s, :, node_idx])
            obs_scores[i, node_idx, 0] = numpy.sum(scale[s:e, node_idx])
            obs_scores[i, node_idx, 1] = tmp + obs_scores[i, node_idx, 0]
        s = obs_indices[start]
        e = obs_indices[end]
        total[s:e, :, node_idx] = numpy.exp(log_total[s:e, :, node_idx] -
                                            numpy.amax(log_total[s:e, :, node_idx],
                                                       axis=1, keepdims=True))
        total[s:e, :, node_idx] /= numpy.sum(total[s:e, :, node_idx], axis=1,
                                             keepdims=True)
        for V in views:
            V.close()
        return

    def __str__(self):
        output = []
        output.append(f"{self.label} model")
        output.append(f"Distributions")
        just = max([len(x.label) for x in self.distributions])
        for D in self.distributions:
            tmp = [D.label.rjust(just)] + [f"{name}:{value}" for name, value in
                               D.get_parameters(log=False).items()]
            output.append(f'  {" ".join(tmp)}')
        just2 = max([len(x.label) for x in self.states])
        output.append("\n\nStates")
        for S in self.states:
            tmp = [S.label.rjust(just2)] + [", ".join([x.label.rjust(just) for x in S.distributions])]
            output.append(f'  {" ".join(tmp)}')
        output.append(f"\n\nInitial Probabilities")
        tmp = [f"{x:0.3f}" for x in self.initial_probabilities.initial_probabilities]
        output.append(", ".join(tmp))
        output.append("")
        output.append(self.transitions.print())
        output = "\n".join(output)
        return output

    def print_distributions(self):
        for D in self.distributions:
            tmp = [D.label] + [f"{name}:{value}" for name, value in
                               D.get_parameters().items()]
            print(" ".join(tmp))
        return

    @classmethod
    def product(cls, X):
        prod = 1
        for x in X:
            prod *= x
        return prod
















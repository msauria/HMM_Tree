import sys
import multiprocessing.managers
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import time

import numpy
import scipy
import matplotlib.pyplot as plt

from .emission import *
from .state import State
from .transition import TransitionMatrix
from .node import Node
from .initial_probability import InitialProbability

class Tree():
    """Base class for hmm model"""

    def __init__(self,
                 tree=None,
                 emissions=None,
                 state_names=None,
                 transition_matrix=None,
                 transition_id=None,
                 initial_probabilities=None,
                 initial_probability_id=None,
                 tree_emissions=None,
                 tree_transition_matrix=None,
                 tree_initial_probabilities=None,
                 RNG=None,
                 nthreads=1,
                 fname=None):
        self.name = "Tree"
        if RNG is None:
            self.RNG = numpy.random.default_rng()
        else:
            self.RNG = RNG
        self.smm_map = None
        self.views = None
        self.num_obs = None
        self.nthreads = max(1, int(nthreads))
        if fname is not None:
            self.load(fname)
            return
        if tree is None or emissions is None:
            raise ValueError("tree and emissions must be passed if model not " \
                             "being loaded from a file")
        self.load_tree(tree)
        assert isinstance(type(emissions), list) or isinstance(type(emissions), dict)
        if isinstance(type(emissions), list):
            if isinstance(type(emissions[0]), list):
                self.num_node_states = len(emissions[0])
                assert len(emissions) == self.nodeN
                node_emissions = emissions
            else:
                assert issubclass(type(emissions[0]), EmissionDistribution)
                self.num_node_states = len(emissions)
                node_emissions = [copy.deepcopy(emissions) for x in range(self.nodeN)]
        else:
            node_emissions = [emissions[x] for x in self.node_order]
        node_transitions = TransitionMatrix(transition_matrix, transition_id)
        if initial_probabilities is not None:
            node_init_probs = InitialProbability(initial_probabilities=initial_probabilities,
                                                 initial_mapping=initial_probability_id,
                                                 seed=self.RNG.rand_int(0, 100000000))
        else:
            node_init_probs = InitialProbability(num_states = self.num_node_states,
                                                 seed=self.RNG.rand_int(0, 100000000))
        for i, name in enumerate(self.node_order):
            node = self.nodes[name]
            node.intialize_HMM(state_names, node_emissions[i], node_init_probs,
                               node_transitions, self.RNG.rand_int(0, 100000000))
        self.num_states = len(tree_emissions)
        self.emissions = tree_emissions
        if tree_initial_probabilities is None:
            self.initial_probabilities = InitialProbability(
                num_states=self.num_states, seed=self.RNG.rand_int(0, 100000000))
        else:
            self.initial_probabilities = InitialProbability(
                initial_probabilities=tree_initial_probabilities)
        self.transitions = TransitionMatrix(tree_transition_matrix)
        self.log_probs = numpy.zeros(self.nodeN + 1, numpy.float64)
        return

    def __str__(self):
        return self.HMM.__str__()

    @classmethod
    def product(cls, X):
        prod = 1
        for x in X:
            prod *= x
        return prod

    def load_tree(self, tree):
        # Evaluate tree
        if type(tree) == str:
            parents = eval(open(tree).read().strip())
        elif type(tree) != dict:
            raise RuntimeError("Incorrect tree format")
        else:
            parents = tree
        # Add nodes and connections
        for c, p in parents.items():
            if p not in self.nodes and p is not None:
                self.node_index[p] = len(self.nodes)
                self.nodes[p] = Node(p, self.node_index[p])
                self.node_tree[self.node_index[p]] = []
            if c not in self.nodes:
                self.node_index[c] = len(self.nodes)
                self.nodes[c] = Node(c, self.node_index[c])
            if p is not None:
                self.nodes[c].parent = self.nodes[p]
                self.nodes[c].parent_name = p
                self.nodes[p].children.append(self.nodes[c])
                self.nodes[p].children_names.append(c)
        # Verify that there is a single root
        for node in self.nodes.values():
            if node.parent is None:
                if self.root is None:
                    self.root = node
                else:
                    raise RuntimeError(f"More than one root found in tree [{self.root},{node.name}]")
        # Label levels in tree nodes
        self.root.find_levels()
        for n in self.nodes.values():
            if n.level is None:
                raise RuntimeError("Not all nodes are in a single tree")
        # Figure out order to evaluate nodes in
        levels = {}
        for name, n in self.nodes.items():
            l = n.level
            levels.setdefault(l, [])
            levels[l].append(name)
        L = list(levels.keys())
        L.sort()
        order = []
        for l in L:
            order += levels[l]
        self.node_order = order
        self.nodeN = len(self.nodes)
        self.node_idx_order = numpy.zeros(self.node_order.shape[0], dtype=numpy.int32)
        self.node_parents = {}
        self.node_children = {}
        self.node_pairs = []
        for i, name in enumerate(self.node_order):
            idx = self.nodes[name].index
            self.node_idx_order[i] = idx
            if self.nodes[name].parent is None:
                self.node_parents[idx] = None
            else:
                self.node_parents[idx] = self.nodes[name].parent.index
                self.node_pairs.append((idx, self.nodes[name].parent.index))
            self.node_children.setdefault(idx, [])
            for child in self.nodes[name].children:
                self.node_children[idx].append(child.index)
        self.node_pairs = numpy.array(self.node_pairs)
        return

    def make_shared_array(self, name, shape, dtype, data=None):
        if name in self.views:
            new_size = ((self.product(shape) * numpy.dtype(dtype).itemsize - 1) //
                        4096 + 1) * 4096
            if self.views[name].size != new_size:
                self.views[name].unlink()
                del self.views[name]
                self.views[name] = self.smm.SharedMemory(self.product(shape) *
                                                    numpy.dtype(dtype).itemsize)
        else:
            self.views[name] = self.smm.SharedMemory(self.product(shape) *
                                                numpy.dtype(dtype).itemsize)
        self.smm_map[name] = self.views[name].name
        new_data = numpy.ndarray(shape, dtype, buffer=self.views[name].buf)
        if data is not None:
            new_data[:] = data
        setattr(self, name, new_data)
        return new_data

    def ingest_observations(self, obs, tree_seqs, obs_mask=None):
        assert self.smm_map is not None, "HmmManager must be used inside a 'with' statement"
        if type(obs) != list:
            obs = [obs]
        for O in obs:
            assert type(O) == numpy.ndarray and O.dtype.names is not None
        self.num_obs = len(obs)
        self.obs_indices = numpy.zeros(self.num_obs + 1, numpy.int64)
        for i in range(len(obs)):
            self.obs_indices[i + 1] = self.obs_indices[i] + obs[i].shape[0]
        self.make_shared_array(f"obs", (self.obs_indices[-1],), obs[0].dtype)
        self.make_shared_array(f"scale", (self.obs_indices[-1], self.nodeN),
                               numpy.float64)
        for i in range(len(obs)):
            s, e = self.obs_indices[i:i+2]
            self.obs[s:e] = obs[i][:]
        self.thread_obs_indices = numpy.round(numpy.linspace(
            0, self.obs.shape[0], self.nthreads + 1)).astype(numpy.int32)
        step = self.obs_indices[-1] / self.nthreads
        self.thread_seq_indices = numpy.searchsorted(
            self.obs_indices, numpy.arange(self.nthreads + 1) * step)
        self.thread_seq_indices[-1] = self.obs_indices.shape[0] - 1
        for i in range(1, self.nthreads):
            target = i * step
            if (target - self.obs_indices[self.thread_seq_indices[i] - 1] <
                self.obs_indices[self.thread_seq_indices[i]] - target):
                self.thread_seq_indices[i] -= 1
        self.make_shared_array(f"obs_scores", (self.num_obs, self.nodeN, 2),
                               numpy.float64)
        assert numpy.amax(tree_seqs) < self.obs_indices[-1]
        self.make_shared_array(f"tree_seqs", (tree_seqs.shape[0],), numpy.int32)
        self.make_shared_array(f"tree_scale", (tree_seqs.shape[0], self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"tree_scores", (tree_seqs.shape[0], 2),
                               numpy.float64)
        self.tree_seqs[:] = tree_seqs
        self.tree_seqN = self.tree_seqs.shape[0]
        self.thread_tree_indices = numpy.round(numpy.linspace(
            0, self.tree_seqN, self.nthreads + 1)).astype(numpy.int32)
        if obs_mask is not None:
            self.make_shared_array(f"obs_mask", (self.obs_indices[-1],
                                   self.num_states), bool)
            for i in range(self.num_obs):
                s, e = self.obs_indices[i:i+2]
                self.obs_mask[s:e, :] = obs_mask[i].astype(bool)

    def train_model(self, obs=None, obs_mask=None, iterations=100, min_delta=1e-8):
        if obs is not None:
            self.ingest_observations(obs, obs_mask)
        else:
            assert self.num_obs is not None
        self.make_shared_array(f"dist_probs",
                               (self.obs_indices[-1], self.root.num_dists, self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"probs",
                               (self.obs_indices[-1], self.num_node_states, self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"forward",
                               (self.obs_indices[-1], self.num_node_states, self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"reverse",
                               (self.obs_indices[-1], self.num_node_states, self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"log_total",
                               (self.obs_indices[-1], self.num_node_states, self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"total",
                               (self.obs_indices[-1], self.num_node_states, self.nodeN),
                               numpy.float64)
        self.make_shared_array(f"tree_probs",
                               (self.tree_seqN, self.num_states, self.nodeN, 5),
                               numpy.float64)
        self.make_shared_array(f"tree_scale",
                               (self.nodeN,), numpy.float64)
        self.make_shared_array(f"tree_weights",
                               (self.tree_seqN, self.num_node_states, self.nodeN),
                               numpy.float64)
        self.tree_weights[:, :, :] = 0
        prev_prob = 1
        for i in range(iterations):
            prob = self.update_model()
            print(f"\r{" "*80}\rIteration {i}: Log-prob {prob: 0.1f}", end='\n',
                  file=sys.stderr)
            #self.plot_params(i)
            # self.save(f"model_iter{i}.npz")
            if abs(prob - prev_prob) / abs(prev_prob) < min_delta:
                break
            prev_prob = prob
        print(f"\r{' ' * 80}\r", end="", file=sys.stderr)
        return

    def update_model(self):
        self.clear_tallies()
        for node in self.nodes:
            self.update_node_model(node)
        self.obs_scores[:, :, 0] -= numpy.amax(self.obs_scores[:, :, 0])
        for node in self.nodes:
            self.update_node_tallies(node)
        self.apply_node_tallies()
        self.find_probs()
        self.find_reverse()
        self.find_forward()
        self.find_total()
        self.update_tallies()
        self.apply_tallies()
        self.find_tree_weights()
        return -self.log_prob[-1]

    def generate_sequences(self, num_seqs=1, lengths=100):
        assert self.smm_map is not None, "HmmManager must be used inside a 'with' statement"
        self.num_obs = int(num_seqs)
        assert type(lengths) == int or self.num_obs == len(lengths)
        if type(lengths) == int:
            lengths = numpy.full(self.num_obs, lengths, numpy.int32)
        elif type(lengths) == list:
            lengths = numpy.array(lengths, numpy.int32)

        self.obs_indices = numpy.zeros(self.num_obs + 1, numpy.int64)
        for i in range(self.num_obs):
            self.obs_indices[i + 1] = self.obs_indices[i] + lengths[i]
        self.make_shared_array(f"obs", (self.obs_indices[-1],),
                               self.HMM.get_emission_dtype())
        self.make_shared_array(f"obs_states", (self.obs_indices[-1],),
                               numpy.int32)
        args = []
        for i in range(self.num_obs):
            args.append([self.obs_indices[i], self.obs_indices[i + 1],
                         self.obs_indices[-1], self.HMM, self.smm_map,
                         self.RNG.integers(0, 99999999)])
        for result in self.pool.starmap(HMM.generate_sequence, args):
            continue
        states = []
        obs = []
        for i in range(self.num_obs):
            s, e = self.obs_indices[i:i+2]
            states.append(numpy.copy(self.obs_states[s:e]))
            obs.append(numpy.copy(self.obs[s:e]))
        return obs, states

    def viterbi(self, obs):
        if type(obs) == numpy.ndarray:
            obs = [obs]
        self.ingest_observations(obs)
        self.make_shared_array(f"dist_probs",
                               (self.obs_indices[-1], self.HMM.num_dists),
                               numpy.float64)
        self.make_shared_array(f"probs",
                               (self.obs_indices[-1], self.HMM.num_states, 4),
                               numpy.float64)
        if self.HMM.num_mixtures > 0:
            self.make_shared_array(f"mix_probs",
                                   (self.obs_indices[-1], self.HMM.num_mixtures),
                                   numpy.float64)
        self.find_probs()
        self.find_paths()
        states = []
        for i in range(self.num_obs):
            s, e = self.obs_indices[i:i+2]
            states.append(self.obs_states[s:e])
        return states, list(self.obs_scores)

    def __enter__(self):
        self.smm_map = {}
        self.views = {}
        self.smm = multiprocessing.managers.SharedMemoryManager()
        self.smm.start()
        self.pool = multiprocessing.Pool(self.nthreads)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()
        self.pool.terminate()
        self.smm.shutdown()
        return

    def clear_tallies(self):
        for node in self.nodes.values():
            node.clear_tallies()
        self.initial_probabilities.clear_tallies()
        self.transitions.clear_tallies()
        for E in self.emissions:
            E.clear_tallies()
        return

    def update_node_model(self, node):
        self.find_node_probs(node)
        self.find_node_forward(node)
        self.find_node_reverse(node)
        self.find_node_total(node)
        self.log_probs[node.index] = numpy.sum(self.obs_scores[:, node.index, 1])
        print(f"{node.label} Logprob: {-self.log_probs[node.index]}")
        return

    def find_node_probs(self, node):
        names = self.obs.dtype.names
        args = []
        for i in range(self.thread_obs_indices.shape[0] - 1):
            s, e = self.thread_obs_indices[i:i+2]
            args.append((s, e, self.obs.dtype, self.probs.shape,
                         self.tree_weights.shape, node, self.smm_map))
        for result in self.pool.starmap(node.find_probs, args):
            continue
        return

    def find_node_forward(self, node):
        args = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            args.append((s, e, self.obs.dtype, self.probs.shape,
                         self.obs_indices, node, self.smm_map))
        for result in self.pool.starmap(node.find_forward, args):
            continue
        return

    def find_node_reverse(self, node):
        args = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            args.append((s, e, self.obs[0].dtype, self.probs.shape,
                         self.obs_indices, node, self.smm_map))
        for result in self.pool.starmap(node.find_reverse, args):
            continue
        return

    def find_node_total(self, node):
        args = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            args.append((s, e, self.obs[0].dtype, self.probs.shape,
                         self.obs_indices, node, self.smm_map))
        for result in self.pool.starmap(node.find_total, args):
            continue
        return

    def update_node_tallies(self, node):
        node.update_tallies(self.probs.shape, self.obs_indices, self.smm_map)
        self.update_node_transition_tallies(node)
        self.update_node_emission_tallies(node)
        return

    def update_node_transition_tallies(self, node):
        args = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            args.append((s, e, node.index, self.probs.shape,
                         node.transitions[node.transitions.valid_trans],
                         node.transitions.valid_trans,
                         self.obs_indices, self.smm_map))
        for result in self.pool.starmap(TransitionMatrix.update_tallies, args):
            node.transitions.tallies += result
        return

    def update_node_emission_tallies(self, node):
        args = []
        for i in range(self.num_node_states):
            for j in range(node.states[i].num_distributions):
                for k in range(self.thread_seq_indices.shape[0] - 1):
                    s, e = self.thread_seq_indices[k:k+2]
                    D = node.states[i].distributions[j]
                    params = D.get_parameters()
                    if D.fixed:
                        continue
                    args.append((s, e, node.index, i, j, None, self.obs.dtype,
                                 self.probs.shape, self.obs_indices,
                                 D.update_tallies, params, self.smm_map))
        for result in self.pool.starmap(EmissionDistribution.update_tallies, args):
            state_idx, dist_idx, _, tallies = result
            node.states[state_idx].distributions[dist_idx].tallies += tallies
        return

    def apply_node_tallies(self):
        for node in self.nodes.values():
            for S in node.states:
                S.apply_tallies()
            node.transitions.apply_tallies()
            node.apply_tallies()
        return

    def find_probs(self):
        args = []
        for i in range(self.thread_tree_indices.shape[0] - 1):
            s, e = self.thread_tree_indices[i:i+2]
            args.append((s, e, self.total.shape, self.tree_probs.shape,
                         self.emissions, self.smm_map))
        for result in self.pool.starmap(self.find_probs_thread, args):
            continue
        return

    def find_reverse(self):
        args = []
        for i in range(self.thread_tree_indices.shape[0] - 1):
            s, e = self.thread_tree_indices[i:i+2]
            args.append((s, e, self.transitions[:, :], self.node_children,
                         self.tree_probs.shape, self.node_idx_order,
                         self.smm_map))
        for result in self.pool.starmap(self.find_reverse_thread, args):
            continue
        return

    def find_forward(self):
        args = []
        for i in range(self.thread_tree_indices.shape[0] - 1):
            s, e = self.thread_tree_indices[i:i+2]
            args.append((s, e, self.initial_probabilities[:], self.transitions[:, :],
                         self.node_parents, self.node_children,
                         self.tree_probs.shape, self.node_idx_order,
                         self.smm_map))
        for result in self.pool.starmap(self.find_forward_thread, args):
            continue
        return

    def find_total(self):
        args = []
        for i in range(self.thread_tree_indices.shape[0] - 1):
            s, e = self.thread_tree_indices[i:i+2]
            args.append((s, e, self.node_children, self.tree_probs.shape,
                         self.node_idx_order, self.smm_map))
        for result in self.pool.starmap(self.find_total_thread, args):
            continue
        return

    def update_tallies(self):
        self.log_probs[-1] = numpy.sum(self.tree_scores[:, 1])
        self.initial_probabilities.update_tallies(numpy.exp(
            self.tree_probs[:, :, 3] + self.tree_scores[:, 0].reshape(-1, 1)))
        self.update_transition_tallies()
        self.update_emission_tallies()
        return

    def update_transition_tallies(self):
        args = []
        for i in range(self.thread_tree_indices.shape[0] - 1):
            s, e = self.thread_tree_indices[i:i+2]
            args.append((s, e, self.node_pairs, self.node_children,
                         self.tree_probs.shape, self.transitions[:, :],
                         self.smm_map))
        for result in self.pool.starmap(TransitionMatrix.update_tree_tallies, args):
            self.transitions.tallies += result
        return

    def update_emission_tallies(self):
        args = []
        for i in range(self.num_states):
            for j in range(self.thread_tree_indices.shape[0] - 1):
                s, e = self.thread_tree_indices[j:j+2]
                D = self.emissions[j]
                else:
                    params = D.get_parameters()
                if D.fixed:
                    continue
                args.append((s, e, i, self.total.shape, self.tree_probs.shape,
                             D.update_tree_tallies, params, self.smm_map))
        for result in self.pool.starmap(EmissionDistribution.update_tree_tallies, args):
            state_idx, tallies = result
            self.emissions[state_idx].tallies += tallies
        return

    def apply_tallies(self):
        for E in self.emissions:
            E.apply_tallies()
        self.transitions.apply_tree_tallies()
        self.initial_probabilities = self.tallies / numpy.sum(self.tallies)
        self.log_initprobs.fill(-numpy.inf)
        where = numpy.where(self.initial_probabilities > 0)[0]
        self.log_initprobs[where] = numpy.log(self.initial_probabilities[where])
        return

    def find_tree_weights(self):
        args = []
        for i in range(self.thread_tree_indices.shape[0] - 1):
            s, e = self.thread_tree_indices[i:i+2]
            args.append((s, e, self.tree_probs.shape, self.emissions, self.smm_map))
        for result in self.pool.starmap(self.find_tree_weights_thread, args):
            continue
        return

    @classmethod
    def find_probs_thread(self, *args):
        start, end, total_shape, probs_shape, emissions, smm_map = args
        obsN, num_states, num_nodes, _ = probs_shape
        views = []
        views.append(SharedMemory(smm_map['total']))
        total = numpy.ndarray(total_shape, numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probs_shape, numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_seqs']))
        tree_seqs = numpy.ndarray((obsN,), numpy.int32, buffer=views[-1].buf)
        for i in range(num_states):
            probs[start:end, :, i, 0] = emissions[i].score_observations(
                total[tree_seqs[start:end], :, :])
        for V in views:
            V.close()
        return

    @classmethod
    def find_reverse_thread(self, *args):
        (start, end, transitions, node_children, probs_shape,
         node_order, smm_map) = args
        seqN, num_states, num_nodes, _ = probs_shape
        views = []
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probs_shape, numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_scale']))
        scale = numpy.ndarray((seqN, num_nodes), numpy.float64, buffer=views[-1].buf)
        for i in node_order[1:][::-1]:
            children_idxs = node_children[i]
            if len(children_idxs) > 0:
                probs[start:end, :, i, 1] = (numpy.sum(
                    probs[start:end, :, children_idxs, 1], axis=2) +
                    probs[start:end, :, i, 1])
            else:
                probs[start:end, :, i, 1] = probs[start:end, :, i, 0]
            probs[start:end, :, i, 1] = scipy.special.logsumexp(
                probs[start:end, :, i, 1].reshape(-1, 1, num_states) + 
                transitions.reshape(1, num_states, num_states), axis=2)
            scale[start:end, i] = numpy.amax(probs[start:end, :, i, 1], axis=1)
            probs[start:end, :, i, 1] -= scale[start:end, i].reshape(-1, 1)
        for V in views:
            V.close()
        return

    @classmethod
    def find_forward_thread(self, *args):
        (start, end, initprobs, transitions, node_parents, node_cihldren,
         probs_shape, node_order, smm_map) = args
        seqN, num_states, num_nodes, _ = probs_shape
        views = []
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probs_shape, numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_scale']))
        scale = numpy.ndarray((seqN, num_nodes), numpy.float64, buffer=views[-1].buf)
        for i in node_order:
            parent_idx = node_parents[i]
            if parent_idx is None:
                probs[start:end, :, i, 2] = (initprobs.reshape(1, -1) +
                                             probs[start:end, :, i, 0])
                scale[start:end, i] = numpy.amax(probs[start:end, :, i, 2])
                probs[start:end, :, i, 2] -= scale[start:end, i].reshape(-1, 1)
            elif len(node_children[parent_idx]) == 1:
                probs[start:end, :, i, 2] = (scipy.special.logsumexp(
                    probs[start:end, :, parent_idx, 2].reshape(-1, num_states, 1) +
                    transitions.reshape(1, num_states, num_states), axis=1) +
                                             probs[start:end, :, i, 0] -
                                             scale[start:end, i].reshape(-1, 1))
            else:
                tmp = probs[start:end, :, parent_idx, 2]
                for j in node_children[parent_idx]:
                    if j == i:
                        continue
                    tmp += probs[start:end, :, j, 1]
                probs[start:end, :, i, 2] = (scipy.special.logsumexp(
                    tmp.reshape(-1, num_states, 1) +
                    transitions.reshape(1, num_states, num_states), axis=1) +
                                             probs[start:end, :, i, 0] -
                                             scale[start:end, i].reshape(-1, 1))
        for V in views:
            V.close()
        return

    @classmethod
    def find_total_thread(self, *args):
        s, e, node_cihldren, probs_shape, node_order, smm_map = args
        seqN, num_states, num_nodes, _ = probs_shape
        views = []
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probs_shape, numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_scores']))
        scores = numpy.ndarray((seqN, 2), numpy.float64, buffer=views[-1].buf)
        for i in node_order:
            children = node_children[i]
            if len(children) == 0:
                probs[s:e, :, i, 3] = probs[s:e, :, i, 2]
            elif len(children) == 1:
                probs[s:e, :, i, 3] = (probs[s:e, :, i, 2] +
                                       probs[s:e, :, children[0], 1])
            else:
                probs[s:e, :, i, 3] = probs[s:e, :, i, 2] + numpy.sum(
                    probs[s:e, :, children, 1], axis=2)

        probs[s:e, :, :, 4] = numpy.exp(probs[s:e, :, :, 3] - numpy.amax(
            probs[s:e, :, :, 3], axis=1, keepdims=True))
        probs[s:e, :, :, 4] /= numpy.sum(probs[s:e, :, :, 4], axis=1,
                                             keepdims=True)
        root = node_order[0]
        scores[s:e, 0] = numpy.sum(tree_scale[s:e, :], axis=1)
        scores[s:e, 1] = (scipy.special.logsumexp(probs[s:e, root, :, 3], axis=1) +
                          scores[s:e, 0])
        for V in views:
            V.close()
        return

    @classmethod
    def find_tree_weights_thread(self, *args):
        start, end, probs_shape, emissions, smm_map = args
        seqN, num_tree_states, num_nodes, _ = probs_shape
        views = []
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probs_shape, numpy.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_weights']))
        tree_weights = numpy.ndarray((seqN, num_node_states), numpy.int32,
                                     buffer=views[-1].buf)
        tree_weights[start:end, :].fill(0)
        for i in range(num_tree_states):
            tree_weights[start:end, :] += (tree_probs[start:end, i, :] *
                                           emissions[i].probabilities.reshape(1, -1))
        tree_weights[start:end, :] /= numpy.sum(tree_probs[start:end, :, :], axis=1)
        tree_weights[start:end, :] = numpy.log(tree_weights[start:end, :])
        for V in views:
            V.close()
        return

    def plot_params(self, iteration):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        N = self.num_states
        M = len([None for x in self.obs.dtype.names if x.count('TSS') == 0])
        hm = numpy.zeros((N, M), dtype=numpy.float64)
        for i in range(N):
            for j in range(M):
                hm[i, j] = self.HMM.states[i].distributions[j].distributions[0].mu
        hm /= numpy.amax(hm, axis=0, keepdims=True)
        ax[0].imshow(hm, aspect='auto')
        for i in range(N):
            for j in range(M):
                hm[i, j] = self.HMM.states[i].distributions[j].distributions[0].sigma
        hm /= numpy.amax(hm, axis=0, keepdims=True)
        ax[1].imshow(hm, aspect='auto')
        plt.savefig(f"param_{iteration}.pdf")
        return

    def save(self, fname):
        data = {}
        data['tree_transition_matrix'] = self.transitions.transition_matrix
        data['tree_initial_probabilities'] = self.initial_probabilities.initial_probabilities
        data['node_transition_matrix'] = self.root.transitions.transition_matrix
        data['node_transition_id'] = self.root.transitions.transition_id
        data['node_initial_probabilities'] = self.root.intitial_probabilities.initial_probabilities
        data['node_initial_id'] = self.root.intitial_probabilities.initial_probability_id
        dist_map = numpy.zeros(self.root.num_dists * self.num_node_states,
            numpy.dtype([('state', numpy.int32), ('dist', numpy.int32),
                         ('index', numpy.int32)]))
        pos = 0
        for i in range(self.num_node_states):
            for j, D in enumerate(self.root.states[i].distributions):
                dist_map[pos] = (i, j, D.index)
                pos += 1
                if f"dist_{D.index}" not in data:
                    params = D.get_parameters()
                    dtype = [('name', f'<U{len(D.name)}'),
                             ('label', f'<U{len(D.label)}')]
                    for name, value in params.items():
                        if (isinstance(type(value), numpy.ndarray) and 
                            (len(value.shape) > 1 or value.shape[0] > 1)):
                            dtype.append((name, numpy.float64,
                                          tuple(list(value.shape) + [self.nodeN])))
                        else:
                            dtype.append((name, numpy.float64, (self.nodeN,)))
                    data[f"dist_{D.index}"] = numpy.zeros(1, dtype=numpy.dtype(dtype))
                    data[f"dist_{D.index}"]['name'] = D.name
                    data[f"dist_{D.index}"]['label'] = D.label
        for i in range(self.num_node_states):
            for nodename in self.node_order:
                node = self.nodes[nodename]
                node_idx = node.index
                for D in node.states[i].distributions:
                    key = f"dist_{D.index}"
                    params = D.get_parameters()
                    for name, value in params.items():
                        if isinstance(type(value), numpy.ndarray):
                            data[key][name][:, node_idx] = value
                        else:
                            data[key][name][node_idx] = value
        data['distribution_mapping'] = dist_map
        state_names = [self.root.states[x].label for x in range(self.num_node_states)]
        data['node_state_names'] = numpy.array(
            state_names, f"<U{max([len(x) for x in state_names])}")
        node_names = ["" for x in range(self.nodeN)]
        for nodename in self.node_order:
            node = self.nodes[nodename]
            node_names[node.index] = node.label
        data['node_names'] = numpy.array(
            node_names, f"<U{max([len(x) for x in node_names])}")
        state_names = [self.emissions[x].label for x in range(len(self.emissions))]
        data['tree_state_names'] = numpy.array(
            state_names, f"<U{max([len(x) for x in state_names])}")
        data['tree_emissions'] = numpy.zeros((self.emissions[0].probabilities.shape[0],
                                              self.num_states), numpy.float64)
        for i, E in enumerate(self.emissions):
            data['tree_emissions'][:, i] = E.probabilities
        pairs = list(self.root.find_pairs().items())
        maxlen = max([len(x[0]) for x in pairs] + [len(x[1]) for x in pairs])
        data['tree'] = numpy.array(pairs, f"<U{maxlen}")
        numpy.savez(fname, **data)
        return

    def load(self, fname):
        temp = numpy.load(fname)
        tree_transition_matrix = temp['tree_transition_matrix']
        node_transition_matrix = temp['node_transition_matrix']
        node_transition_id = temp['node_transition_id']
        tree_initial_probabilities = temp['tree_initial_probabilities']
        node_initial_probabilities = temp['node_initial_probabilities']
        node_initial_probability_id = temp['node_initial_probability_id']
        tree_state_names = temp['tree_state_names']
        node_state_names = temp['node_state_names']
        node_names = temp['node_names']
        tree = {x[0]: x[1] for x in temp['tree']}
        tree_num_states = temp['tree_emissions'].shape[0]
        tree_emissions = [EmissionAlphabetDistribution(temp['tree_emissions'][x, :],
                                                       label=tree_state_names[x])
                          for x in range(tree_num_states)]
        dist_map = temp['distribution_mapping']
        distributions = {}
        node_num_states = len(node_state_names)
        dists = {
            "Base": EmissionDistribution,
            "Alphabet": EmissionAlphabetDistribution,
            "Poisson": EmissionPoissonDistribution,
            "DiscreteMixture": EmissionDiscreteMixtureDistribution,
            "Gaussian": EmissionGaussianDistribution,
            "LogNormal": EmissionLogNormalDistribution,
            "Gamma": EmissionGammaDistribution,
            "Zero": EmissionZeroDistribution,
            "ContinuousMixture": EmissionContinuousMixtureDistribution,
        }
        num_nodes = len(node_names)
        for j in range(num_nodes):
            for name in temp.keys():
                if not name.startswith('dist_'):
                    continue
                params = {}
                for name2 in temp[name].dtype.names:
                    if name2 == 'name':
                        dname = temp[name][name2][0]
                    elif name2 == 'label':
                        label = temp[name][name2][0]
                    else:
                        if temp[name][name2].size == 1:
                            params[name2] = temp[name][name2][j]
                        else:
                            params[name2] = temp[name][name2][:, j]
                if dname == "Zero":
                    distributions[name] = dists[dname]()
                elif not dname.endswith("Mixture"):
                    distributions[name] = dists[dname](**params)
                # else:
                #     index = numpy.where(dist_map['index'] == int(name.split('_')[-1]))[0][0]
                #     indices = dist_map['index'][numpy.where(numpy.logical_and(numpy.logical_and(
                #         dist_map['state'] == dist_map['state'][index],
                #         dist_map['dist'] == dist_map['dist'][index]),
                #         dist_map['mixdist'] != -1))]
                #     distributions[name] = dists[dname](
                #         [mixdistributions[f"mixdist_{x}"] for x in indices],
                #         **params)
                distributions[f"{name}_{j}"].label = label
        emissions = [[[None for y in range(numpy.amax(dist_map['dist']) + 1)]
                     for x in range(node_num_states)] for z in range(num_nodes)]
        for i in range(num_nodes):
            for s_idx, d_idx, m_idx, idx in dist_map:
                if m_idx == -1:
                    emissions[i][s_idx][d_idx] = distributions[f"dist_{idx}_{i}"]
        self.load_tree(tree)
        self.num_node_states = node_num_states
        node_transitions = TransitionMatrix(node_transition_matrix, node_transition_id)
        node_init_probs = InitialProbability(initial_probabilities=node_initial_probabilities,
                                             initial_mapping=node_initial_probability_id)
        for i, name in enumerate(self.node_order):
            node = self.nodes[name]
            node.intialize_HMM(node_state_names, emissions[i], node_init_probs,
                               node_transitions)
        self.num_states = tree_num_states
        self.emissions = tree_emissions
        self.initial_probabilities = InitialProbability(
            initial_probabilities=tree_initial_probabilities)
        self.transitions = TransitionMatrix(tree_transition_matrix)
        self.log_probs = numpy.zeros(self.nodeN + 1, numpy.float64)
        return











from multiprocessing.shared_memory import SharedMemory

import numpy


class TransitionMatrix():
    """Base class for holding hmm transition probabilities"""

    def __init__(self, num_states=None, transition_matrix=None, transition_id=None, RNG=numpy.random):
        assert (not num_states is None) != (not transition_matrix is None)
        assert ((transition_id is None) or
                ((transition_matrix is not None) and (transition_id.shape == transition_matrix.shape)) or
                (transition_id.shape == (num_states, num_states)))
        if not num_states is None:
            self.transition_matrix = numpy.zeros((int(num_states), int(num_states)), numpy.float64)
            self.transition_matrix = RNG.random(size=self.transition_matrix.shape)
            self.update_mask = numpy.ones(self.transition_matrix.shape, bool)
        else:
            self.transition_matrix = numpy.copy(transition_matrix).astype(numpy.float64)
        self.num_states = self.transition_matrix.shape[0]
        if transition_id is not None:
            self.transition_id = numpy.copy(transition_id).astype(numpy.int32)
            self.transition_matrix[numpy.where(self.transition_id < 0)] = 0
        else:
            self.transition_id = numpy.arange(self.num_states ** 2).reshape(self.num_states, -1)
        self.num_transitions = numpy.amax(self.transition_id) + 1
        where = numpy.where(self.transition_id < 0)
        self.transition_matrix[where] = 0
        self.transition_matrix /= numpy.sum(self.transition_matrix, axis=1, keepdims=True)
        self.log_transition_matrix = numpy.zeros_like(self.transition_matrix)
        self.log_transition_matrix.fill(-numpy.inf)
        where = numpy.where(self.transition_matrix > 0)
        self.log_transition_matrix[where] = numpy.log(self.transition_matrix[where])
        self.valid_trans = numpy.where(self.transition_id >= 0)
        self.tallies = numpy.zeros(self.valid_trans[0].shape[0], numpy.float64)
        return

    def __getitem__(self, idx):
        return self.log_transition_matrix[idx]

    def __setitem__(self, idx, value):
        self.transition_matrix[idx] = value
        return

    def clear_tallies(self):
        self.tallies[:] = 0
        self.updated = False
        return

    @classmethod
    def update_tallies(self, *args):
        start, end, idx, probsShape, transitions, valid_trans, obs_indices, smm_map = args
        obsN, stateN, num_nodes = probsShape
        views = []
        views.append(SharedMemory(smm_map['probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['forward']))
        forward = numpy.ndarray(probsShape, dtype=numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['reverse']))
        reverse = numpy.ndarray(probsShape, dtype=numpy.float64,
                                buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['scale']))
        scale = numpy.ndarray((obsN, num_nodes), dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_scores']))
        obs_scores = numpy.ndarray((obsN, num_nodes, 2), dtype=numpy.float64,
                                   buffer=views[-1].buf)
        tallies = numpy.zeros(valid_trans[0].shape[0], numpy.float64)
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            xi = numpy.exp(forward[s:e - 1, valid_trans[0], idx] +
                           reverse[s + 1:e, valid_trans[1], idx] +
                           probs[s + 1:e, valid_trans[1], idx] +
                           transitions.reshape(1, -1) + obs_scores[i, idx, 0])
            tallies += numpy.sum(xi, axis=0)
        for V in views:
            V.close()
        return tallies

    @classmethod
    def update_tree_tallies(self, *args):
        s, e, pairs, node_children, probsShape, transitions, smm_map = args
        seqN, stateN, num_nodes, _ = probsShape[:2]
        views = []
        views.append(SharedMemory(smm_map['tree_probs']))
        probs = numpy.ndarray(probsShape, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_scale']))
        scale = numpy.ndarray(obsN, dtype=numpy.float64,
                              buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['tree_scores']))
        scores = numpy.ndarray((seqN, 2), dtype=numpy.float64, buffer=views[-1].buf)
        tallies = numpy.zeros((num_states, num_states), numpy.float64)
        for idx1, idx2 in pairs:
            children = node_children[idx2]
            if len(children) == 0:
                reverse = probs[s:e, :, idx2, 0] - scale[s:e, idx2]
            elif len(children) == 1:
                reverse = (probs[s:e, :, idx2, 0] - scale[s:e, idx2] +
                           probs[s:e, :, children[0], 1])
            else:
                reverse = (probs[s:e, :, idx2, 0] - scale[s:e, idx2] +
                           numpy.sum(probs[s:e, :, children, 1], axis=2))
            xi = numpy.exp(probs[s:e, :, idx1, 2].reshape(-1, num_states, 1) +
                           reverse.reshape(-1, 1, num_states) +
                           transitions.reshape(1, num_states, num_states) +
                           scores[s:e, 0].reshape(-1, 1, 1))
            tallies += numpy.sum(xi, axis=0)
        for V in views:
            V.close()
        return tallies

    def apply_tallies(self):
        if self.updated:
            return
        self.updated = True
        tallies = numpy.bincount(self.transition_id[self.valid_trans],
                                 weights=self.tallies,
                                 minlength=self.num_transitions)
        tallies /= numpy.maximum(1, numpy.bincount(self.transition_id[self.valid_trans],
                                                   minlength=self.num_transitions))
        self.transition_matrix.fill(0)
        self.transition_matrix[self.valid_trans] = tallies[
            self.transition_id[self.valid_trans]]
        self.transition_matrix /= numpy.sum(self.transition_matrix, axis=1,
                                            keepdims=True)
        where = numpy.where(self.transition_matrix > 0)
        self.log_transition_matrix.fill(-numpy.inf)
        self.log_transition_matrix[where] = numpy.log(self.transition_matrix[where])
        return

    def apply_tree_tallies(self):
        if self.updated:
            return
        self.updated = True
        self.transition_matrix[:, :] = tallies
        self.transition_matrix /= numpy.sum(self.transition_matrix, axis=1,
                                            keepdims=True)
        where = numpy.where(self.transition_matrix > 0)
        self.log_transition_matrix.fill(-numpy.inf)
        self.log_transition_matrix[where] = numpy.log(self.transition_matrix[where])
        return

    def print(self):
        output = []
        output.append("Transitions")
        tmp = " ".join([f"{x}".rjust(6, ' ') for x in range(self.num_states)])
        output.append(f" State {tmp}")
        for i in range(self.num_states):
            tmp = " ".join([f"{x*100:0.1f}%".rjust(6, ' ') for x in self.transition_matrix[i, :]])
            state = f"{i}".rjust(5, ' ')
            output.append(f" {state} {tmp}")
        output = "\n".join(output)
        return output

    def generate_transition(self, state, RNG):
        return numpy.searchsorted(numpy.cumsum(
            self.transition_matrix[state, :]), RNG.random())












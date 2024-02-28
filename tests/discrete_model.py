#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import HMM

def main():
    RNG = numpy.random.default_rng(seed=20103)
    # define 2 dice distributions
    fair = HMM.EmissionAlphabetDistribution(probabilities=numpy.array([0.133, 0.133, 0.133, 0.133, 0.133, 0.133], numpy.float64))
    unfair = HMM.EmissionAlphabetDistribution(probabilities=numpy.array([0.01,0.01,0.01, 0.14, 0.14, 0.69], numpy.float64))
    # define transitions
    TM = numpy.array([[0.9, 0.1], [0.05, 0.95]], numpy.float64)
    TMb = numpy.array([[0.8, 0.2], [0.1, 0.9]], numpy.float64)
    # define initial probabilities
    IP = numpy.array([0.4, 0.6], numpy.float64)
    # define model
    with HMM.HmmManager(num_states=2, emissions=[fair, unfair],
                        transition_matrix=TM, initial_probabilities=IP, RNG=RNG) as model, \
        HMM.HmmManager(num_states=2, emissions=[
            HMM.EmissionAlphabetDistribution(alphabet_size=6, RNG=RNG),
            HMM.EmissionAlphabetDistribution(alphabet_size=6, RNG=RNG)],
            #fair, unfair],
            transition_matrix=TMb,
            #initial_probabilities=IP,
            RNG=RNG,
            nthreads=8) as naive_model:
        # Generate observations/states
        obs, states = model.generate_sequences(num_seqs=100, lengths=100)
        print(model)
        # Train randomly initialized model
        # vit = model.viterbi(obs[0])[0]
        # model.train_model(obs, iterations=1)
        # for i in range(30):#obs[0].shape[0]):
        #     print(obs[0]["0"][i],states[0][i], vit[i], model.probs[i, :], model.forward[i, :], model.reverse[i, :], model.total[i, :])
        # print(numpy.sum(states[0] == vit)/vit.shape[0], numpy.bincount(states[0]), numpy.bincount(vit))
        naive_model.train_model(obs, iterations=100)
        print(naive_model)
        # vit = naive_model.viterbi(obs[0])[0]
        # for i in range(30):#obs[0].shape[0]):
        #     print(obs[0]["0"][i], states[0][i], numpy.argmax(naive_model.total[i, :]), naive_model.probs[i, :], naive_model.forward[i, :], naive_model.reverse[i, :], naive_model.total[i, :])
        # print(numpy.sum(states[0] == vit)/vit.shape[0], numpy.bincount(states[0]), numpy.bincount(vit))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
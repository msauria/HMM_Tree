#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import HMM

def main():
    RNG = numpy.random.default_rng(seed=20106)
    # define 2 distributions
    fair = HMM.EmissionGaussianDistribution(mu=0.0, sigma=1.0)
    unfair = HMM.EmissionGaussianDistribution(mu=2.0, sigma=2.25)
    # define transitions
    TM = numpy.array([[0.9, 0.1], [0.05, 0.95]], numpy.float64)
    # define initial probabilities
    IP = numpy.array([0.4, 0.6], numpy.float64)
    # define model
    with HMM.HmmManager(num_states=2, emissions=[fair, unfair],
                        transition_matrix=TM, initial_probabilities=IP,
                        RNG=RNG, nthreads=8) as model:
        # Generate observations/states
        obs, states = model.generate_sequences(num_seqs=100, lengths=200)
        print(model)
        # transitions = numpy.zeros((2, 2), numpy.int32)
        # means = numpy.zeros(2, numpy.float64)
        # var = numpy.zeros(2, numpy.float64)
        # counts = numpy.zeros(2, numpy.int32)
        # for i in range(len(obs)):
        #     transitions[0, 0] += numpy.sum(numpy.logical_and(states[i][:-1] == 0, states[i][1:] == 0))
        #     transitions[0, 1] += numpy.sum(numpy.logical_and(states[i][:-1] == 0, states[i][1:] == 1))
        #     transitions[1, 0] += numpy.sum(numpy.logical_and(states[i][:-1] == 1, states[i][1:] == 0))
        #     transitions[1, 1] += numpy.sum(numpy.logical_and(states[i][:-1] == 1, states[i][1:] == 1))
        #     means[0] += numpy.sum(obs[i]['0'][numpy.where(states[i] == 0)])
        #     means[1] += numpy.sum(obs[i]['0'][numpy.where(states[i] == 1)])
        #     var[0] += numpy.sum(obs[i]['0'][numpy.where(states[i] == 0)]**2)
        #     var[1] += numpy.sum(obs[i]['0'][numpy.where(states[i] == 1)]**2)
        #     counts[0] += numpy.where(states[i] == 0)[0].shape[0]
        #     counts[1] += numpy.where(states[i] == 1)[0].shape[0]
        # transitions = transitions / numpy.sum(transitions, axis=1, keepdims=True)
        # means /= counts
        # var = (var / counts - means ** 2) ** 0.5
        # print(transitions)
        # print("Means", means)
        # print("Sigma", var)
        # # Train randomly initialized model
        # vit = model.viterbi(obs)[0]
        # matched = 0
        # total = 0
        # proportions = numpy.zeros(2, numpy.int32)
        # vproportions = numpy.zeros(2, numpy.int32)
        # model.train_model(obs, iterations=1)
        # for i in range(min(30, obs[0].shape[0])):
        #     print(obs[0]["0"][i],states[0][i], vit[0][i], model.probs[i, :, 0], model.probs[i, :, 1], model.probs[i, :, 2], model.probs[i, :, 3])
        # for i in range(len(obs)):
        #     matched += numpy.sum(states[i] == vit[i])
        #     total += vit[i].shape[0]
        #     proportions += numpy.bincount(states[i], minlength=2)
        #     vproportions += numpy.bincount(vit[i], minlength=2)
        # print(matched/total, proportions, vproportions)

    with HMM.HmmManager(num_states=2, emissions=[
            HMM.EmissionGaussianDistribution(mu=0, sigma=1.0),
            HMM.EmissionGaussianDistribution(mu=2, sigma=1.0)],
            RNG=RNG, nthreads=8) as naive_model:
        naive_model.train_model(obs, iterations=100)
        print(naive_model)
        # vit = naive_model.viterbi(obs)[0]
        # naive_model.train_model(obs, iterations=1)
        # for i in range(min(30, obs[0].shape[0])):
        #     print(obs[0]["0"][i],states[0][i], vit[0][i], naive_model.probs[i, :, 0], naive_model.probs[i, :, 1], naive_model.probs[i, :, 2], naive_model.probs[i, :, 3])
        # matched = 0
        # total = 0
        # proportions = numpy.zeros(2, numpy.int32)
        # vproportions = numpy.zeros(2, numpy.int32)
        # for i in range(len(obs)):
        #     matched += numpy.sum(states[i] == vit[i])
        #     total += vit[i].shape[0]
        #     proportions += numpy.bincount(states[i], minlength=2)
        #     vproportions += numpy.bincount(vit[i], minlength=2)
        # print(matched/total, proportions, vproportions)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
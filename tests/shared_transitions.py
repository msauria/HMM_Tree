#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import HMM

def main():
    RNG = numpy.random.default_rng(seed=20106)
    # define 2 distributions
    fair1 = HMM.EmissionGaussianDistribution(mu=-2.0, sigma=1.0)
    fair2 = HMM.EmissionGaussianDistribution(mu=2.0, sigma=1.0)
    unfair1 = HMM.EmissionGaussianDistribution(mu=5.0, sigma=2.25)
    unfair2 = HMM.EmissionGaussianDistribution(mu=15.0, sigma=2.25)
    fair = HMM.EmissionContinuousMixtureDistribution(
        distributions=[fair1, fair2], proportions=[0.33, 0.67])
    unfair = HMM.EmissionContinuousMixtureDistribution(
        distributions=[unfair1, unfair2], proportions=[0.75, 0.25])
    # define transitions
    TM = numpy.array([[0.9, 0.1], [0.1, 0.9]], numpy.float64)
    Tid = numpy.array([[0, 1], [1, 0]], numpy.int32)
    # define initial probabilities
    IP = numpy.array([0.4, 0.6], numpy.float64)
    # define model
    with HMM.HmmManager(num_states=2, emissions=[fair, unfair],
                        transition_matrix=TM, transition_id=Tid,
                        initial_probabilities=IP,
                        RNG=RNG, nthreads=8) as model:
        # Generate observations/states
        print(model)
        obs, states = model.generate_sequences(num_seqs=50, lengths=500)
        model.train_model(obs, iterations=1)

    fair1b = HMM.EmissionGaussianDistribution(mu=0.0, sigma=1.0)
    fair2b = HMM.EmissionGaussianDistribution(mu=1.0, sigma=1.0)
    unfair1b = HMM.EmissionGaussianDistribution(mu=9.0, sigma=1)
    unfair2b = HMM.EmissionGaussianDistribution(mu=10.0, sigma=1)
    fairb = HMM.EmissionContinuousMixtureDistribution(
        distributions=[fair1b, fair2b], proportions=[0.33, 0.67])
    unfairb = HMM.EmissionContinuousMixtureDistribution(
        distributions=[unfair1b, unfair2b], proportions=[0.75, 0.25])
    TMb = numpy.array([[0.6, 0.4], [0.2, 0.8]], numpy.float64)
    IPb = numpy.array([0.5, 0.5], numpy.float64)
    with HMM.HmmManager(num_states=2, emissions=[fairb, unfairb],
            RNG=RNG, transition_matrix=TMb, transition_id=Tid,
            initial_probabilities=IPb, nthreads=8) as naive_model:
        print(naive_model)
        naive_model.train_model(obs, iterations=30)
        print(naive_model)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
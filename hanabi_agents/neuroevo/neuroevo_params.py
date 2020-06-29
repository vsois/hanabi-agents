"""Parametrization for a neuroevolutionary population"""
from typing import NamedTuple, List
import gin

@gin.configurable
class NeuroEvoParams(NamedTuple):
    """Parametrization for a neuroevolutionary population

    Internal variables:
        population_size -- Number of neuroevo agents.
        chromosome_init_layers -- Number of layers with which new chromosomes are initiated.
        chromosome_n_seeds -- Number of seeds in each chromosome.
        crossover_attempts -- Number of times to attempt chrossover.
        extinction_period -- How often extinction happens. During extinction only n_survivors most fit agents survive.
        n_survivors -- Number of surviving agents during extinction event.
        seed -- random seed.
    """
    population_size: int
    chromosome_init_layers: List[int]
    chromosome_n_seeds: int
    #  fitness: Callable[[Observations, Actions, Rewards], List[float]]
    #  mutation: Mutation
    #  crossover: Crossover
    crossover_attempts: int
    extinction_period: int
    n_survivors: int
    seed: int = 42

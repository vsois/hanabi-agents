"""Crossover -- a way to recombine two chromosomes"""
from typing import List
import numpy as np
from .chromosome import Chromosome

class Crossover:
    """Recombine two chromosomes"""
    def __init__(self, crossover_proba: float = 0.5):
        self.proba = crossover_proba

    def chance_crossover(self) -> bool:
        """Should do crossover?"""
        return np.random.uniform() < self.proba

    @staticmethod
    def crossover(chromosome1: Chromosome, chromosome2: Chromosome) -> List[Chromosome]:
        """Crossover two chomosomes and get a new one. (this version splits in half)"""
        half_seeds_len = len(chromosome1.seeds) // 2
        half_layers_len1 = len(chromosome1.layer_sizes) // 2
        half_layers_len2 = len(chromosome2.layer_sizes) // 2
        new_chromosomes = [Chromosome(), Chromosome()]
        new_chromosomes[0].seeds = chromosome1.seeds[:half_seeds_len] \
                                 + chromosome2.seeds[half_seeds_len:]
        new_chromosomes[0].layer_sizes = chromosome1.layer_sizes[:half_layers_len1] \
                                       + chromosome2.layer_sizes[half_layers_len2 + 1:]
        new_chromosomes[1].seeds = chromosome2.seeds[:half_seeds_len] \
                                 + chromosome1.seeds[half_seeds_len:]
        new_chromosomes[1].layer_sizes = chromosome2.layer_sizes[:half_layers_len2] \
                                       + chromosome1.layer_sizes[half_layers_len1 + 1:]
        return new_chromosomes

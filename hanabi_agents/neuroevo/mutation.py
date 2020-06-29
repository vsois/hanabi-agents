"""Mechanism for altering chromosomes."""
import numpy as np
from .chromosome import Chromosome

class Mutation:
    """Mutation class mutates chromosomes based on fixed mutation probabilities"""
    def __init__(
            self,
            seed_mutation_proba,
            layer_size_mutation_proba,
            layer_number_mutation_proba):
        self.seed_mutation_proba = seed_mutation_proba
        self.layer_size_mutation_proba = layer_size_mutation_proba
        self.layer_number_mutation_proba = layer_number_mutation_proba

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Mutate chromosome according to the mutation probabilities"""
        keys = np.random.uniform(size=len(chromosome) + 2) # + 2 for the case if we add a layer
        layer_number_key = keys[0]
        if layer_number_key < self.layer_number_mutation_proba:
            if (self.layer_number_mutation_proba / 2 - layer_number_key) < 0:
                chromosome.layer_sizes.pop()
            else:
                chromosome.layer_sizes.append(chromosome.layer_sizes[-1])

        n_layers = len(chromosome.layer_sizes)
        for i in range(n_layers):
            layer_size_key = keys[i + 1]
            if layer_size_key < self.layer_size_mutation_proba:
                if (self.layer_size_mutation_proba / 2 - layer_size_key) < 0:
                    chromosome.layer_sizes[i] /= 2
                else:
                    chromosome.layer_sizes[i] *= 2

        for i in range(len(chromosome.seeds)):
            seed_key = keys[i + n_layers + 1]
            if seed_key < self.seed_mutation_proba:
                if (self.seed_mutation_proba / 2 - seed_key) < 0:
                    chromosome.seeds[i] -= 1
                else:
                    chromosome.seeds[i] += 1

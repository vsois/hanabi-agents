"""Container representing a chromosome"""
from typing import NamedTuple, List

class Chromosome(NamedTuple):
    """Chromosome represents the mutable part of the neuroevolunary agent.

    Arguments:
        seeds -- Sequence of seeds used for weight generation.
                 Seeds are mutable, but their number remains constant.
        layer_sizes -- Sized of network layers.
                       Both number of layers and their sizes are mutable.

    Functions:
        __len__ -- Legth of the chromosome which is number of layers and number of seeds.
    """
    seeds: List[int]
    layer_sizes: List[int]

    def __len__(self):
        return len(self.seeds + self.layer_sizes)

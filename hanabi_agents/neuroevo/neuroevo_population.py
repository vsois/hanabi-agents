"""
A neuroevolutionary algorithm to train a population of agents
"""
from typing import Callable
import numpy as np
import haiku as hk
import jax
from jax import numpy as jnp
import gin
from .chromosome import Chromosome
from .crossover import Crossover
from .mutation import Mutation
from .neuroevo_params import NeuroEvoParams
from .neuroevo_agent import NeuroEvoAgent

Actions = np.ndarray
Observations = np.ndarray
Rewards = np.ndarray
LegalMoves = np.ndarray


class NeuroEvoPopulation:
    """An agent which maintains and trains a population of networks.
    """

    def __init__(
            self,
            observation_spec,
            action_spec,
            fitness: Callable,
            mutation: Mutation,
            crossover: Crossover,
            params: NeuroEvoParams):
        self.params = params
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(params.seed))
        self.n_states = observation_spec.shape[0]
        self.observation_len = observation_spec.shape[1]
        self.n_actions = action_spec.num_values
        assert (self.n_states % params.population_size) == 0, \
                "number of states must be a multiple of population size"
        self.pop_size = params.population_size
        self.fitness = fitness
        self.crossover = crossover
        self.crossover_attempts = params.crossover_attempts
        self.mutation = mutation
        self.extinction_period = params.extinction_period
        self.n_survivors = params.n_survivors
        self.states_done = np.full((self.n_states,), False)
        self.states_reset = np.full((self.n_states,), False)
        self.evaluations = np.zeros((self.n_states,))

        self.agents = []
        for _ in range(self.params.population_size):
            chromosome = Chromosome(
                seeds=np.random.randint(0, 1e10, size=params.chromosome_n_seeds),
                layer_sizes=params.chromosome_init_layers)
            agent = NeuroEvoAgent(
                self.observation_len,
                self.n_actions,
                self.n_states // self.pop_size,
                chromosome)
            self.agents.append(agent)

        self.evo_steps = 0

    def explore(self, observations):
        """Same as exploit"""
        self.exploit(observations[1][0])

    def exploit(self, observations: Observations):
        """Let specimen generate actions"""
        actions = []
        obs_chunk_size = self.params.n_states // self.params.pop_size
        for i, agent in enumerate(self.agents):
            actions.append(agent.exploit(
                observations[i * obs_chunk_size : (i + 1) * obs_chunk_size]))
        return np.concatenate(actions, axis=0)

    def evaluation_done(self):
        """reached the end in all games?"""
        return np.all(self.states_done)

    def add_experience_first(self, observations: Observations, step_types):
        """Record states which were reset"""
        self.states_reset[step_types == 0] = True

    def add_experience(
            self,
            observations: Observations,
            actions: Actions,
            rewards: Rewards,
            step_types):
        """Evaluate finess"""
        working_states = np.logical_not(self.states_done)
        working_states = np.logical_and(working_states, self.states_reset)
        self.evaluations[working_states] += self.fitness(
            observations,
            actions,
            rewards,
            [a.chromosome for a in self.agents])[working_states]
        self.states_done[np.logical_and(step_types == 2, working_states)] = True

    def update(self):
        """Perform evolutionary step"""
        if not self.evaluation_done():
            return

        # extinct
        if self.evo_steps % self.params.extinction_period:
            fit_threshold = sorted(self.evaluations, reverse=True)[self.params.n_survivors]
            self.agents = self.agents[self.evaluations > fit_threshold]
            self.agents = self.agents * (self.params.pop_size // self.params.n_survivors)

        chromosomes = [a.chromosome for a in self.agents]
        # crossover
        for _ in range(self.params.crossover_attempts):
            if self.crossover.chance_crossover():
                # based on fitness?
                chr_idx = np.random.randint(0, self.params.pop_size, 2)
                recombined = self.crossover.crossover(chromosomes[chr_idx[0]],
                                                      chromosomes[chr_idx[1]])
                chromosomes[chr_idx[0]] = recombined[0]
                chromosomes[chr_idx[1]] = recombined[1]


        # mutate
        chromosomes = [self.mutation.mutate(c) for c in chromosomes]

        # apply mutations
        for chromosome, agent in zip(chromosomes, self.agents):
            agent.chromosome = chromosome


        self.states_done[:] = False
        self.states_reset[:] = False
        self.evaluations[:] = 0
        self.evo_steps += 1

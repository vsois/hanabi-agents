"""An agent trainable using neuroenvolutionary algorithm"""
from typing import List, Tuple
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
from .chromosome import Chromosome

Observations = np.ndarray
LegalMoves = np.ndarray
Actions = np.ndarray

def build_mlp_network(arch: List[int], num_actions: int) -> hk.Transformed:
    """Build an MLP"""

    def q_net(obs):
        arch = tuple(arch) + (num_actions, )
        network = hk.nets.MLP(arch)
        return network(obs)

    return hk.transform(q_net)

def build_lstm_network(arch: List[int], num_actions: int) -> hk.Transformed:
    """Builds the lstm network."""
    def q_net(obs: Observations, state: hk.LSTMState) -> Tuple[Actions, hk.LSTMState]:
        layers = []
        for i in range(len(arch) - 1):
            layers.extend([hk.LSTM(arch[i]), jax.nn.relu])
        layers.append(hk.LSTM(arch[-1]))
        # readout
        layers.append(hk.nets.MLP([arch[-1], num_actions]))
        network = hk.DeepRNN(layers)
        return network(obs, state)

    def state_initializer(hidden_sizes: List[int], batch_size: int) -> hk.LSTMState:
        states = []
        for hidden_size in hidden_sizes:
            state = hk.LSTMState(hidden=jnp.zeros([hidden_size]),
                                 cell=jnp.zeros([hidden_size]))
            broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
            state = jax.tree_map(broadcast, state)
            states.append(state)
        return tuple(states)

    return state_initializer, hk.transform(q_net)


class NeuroEvoAgent:
    """Represents one speciman of the population.
    """
    def __init__(
            self,
            observation_len: int,
            num_actions: int,
            n_states: int,
            #  layers: List[int]
            chromosome: Chromosome
        ):

        self.observation_len = observation_len
        self.num_actions = num_actions
        self.n_states = n_states
        self.chromosome = chromosome
        #  self.generate_network()
        #  generate_weights(self.init_params, )

    #  @jax.jit
    @staticmethod
    def generate_weights(initial_weights: hk.Params, seeds: List[int]):
        """Generate weights by adding random values generated from the provided
        seeds to the initial weights.
        """
        values, tree_def = jax.tree_util.tree_flatten(initial_weights)
        for i, val in enumerate(values):
            #  lax.fori_loop(0, len(values), body_fun, init_val)
            for seed in seeds:
                values[i] = jnp.add(val, jax.random.normal(jax.random.PRNGKey(seed), val.shape))

        return jax.tree_util.tree_unflatten(tree_def, values)

    def generate_network(self):
        state_initializer, network = build_lstm_network(self.chromosome.layer_sizes, self.num_actions)
        self.state = state_initializer(self.chromosome.layer_sizes, self.n_states)
        #  print(self.state)
        self.init_params = network.init(
            jax.random.PRNGKey(0),
            np.ones((self.n_states, self.observation_len)),
            self.state
        )
        #  self.init_params = network.init(hk.initializers.Constant(0),
        #                                  np.ones((n_states, observation_len)))

        def policy(net_params: hk.Params, observations: Observations, legal_moves: LegalMoves):
            vals, self.state = network.apply(net_params, observations, self.state)
            return jnp.argmax(vals)

        self.policy = jax.jit(policy)
        #  self.policy = policy

    @property.setter
    def chromosome(self, new_chromosome):
        old_chromosome = self.chromosome
        self.chromosome = new_chromosome
        if old_chromosome is None or old_chromosome.layer_sizes != new_chromosome.layer_sizes:
            self.generate_network()
        self.net_params = generate_weights(self.init_params, self.chromosome.seeds)

    def exploit(self, observations: Observations, legal_moves: LegalMoves):
        """exploit the policy"""
        return self.policy(self.net_params, observations, legal_moves)

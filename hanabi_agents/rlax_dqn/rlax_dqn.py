"""
This file implements a DQNAgent.
"""
import collections
from functools import partial
from typing import Sequence, Tuple, List

import haiku as hk
from haiku import nets
import jax
from jax.experimental import optix
import jax.numpy as jnp
import rlax


Flags = collections.namedtuple(
        "Flags",
        "layers, epsilon, temperature, discount_factor, learning_rate, seed",
        defaults=[[50], 0.25, 1., 0.99, 0.001, 1234])
FLAGS = Flags()

DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy"])


def build_network(layers: list, num_actions: int) -> hk.Transformed:

    def q_net(obs):
        layers_ = tuple(layers) + (num_actions, )
        network = hk.nets.MLP(layers_)
        return network(obs)

    return hk.transform(q_net)


class DQNPolicy:
    """greedy and epsilon-greedy policies for DQN agent"""

    @staticmethod
    def _categorical_sample(key, probs):
        """Sample from a set of discrete probabilities."""
        cpi = jnp.cumsum(probs, axis=-1)
        # TODO
        # sometimes illegal actions emerge due to numerical inaccuracy.
        # e.g. 2 actions, last action 100%: -> cpi = [0, 1]
        # but due to numerical stuff: cpi = [0, 0,997]
        # sample rnd = 0.999 -> rnd > cpi = [T, T] -> argmin returns 0 instead of 1
        cpi = jax.ops.index_update(cpi, jax.ops.index[:, -1], 1.)
        rnds = jax.random.uniform(key, shape=probs.shape[:-1] + (1,), maxval=0.999)
        return jnp.argmin(rnds > cpi, axis=-1)



    @staticmethod
    def _mix_with_legal_uniform(probs, epsilon, legal):
        """Mix an arbitrary categorical distribution with a uniform distribution."""
        num_legal = jnp.sum(legal, axis=-1, keepdims=True)
        uniform_probs = legal / num_legal
        return (1 - epsilon) * probs + epsilon * uniform_probs

    @staticmethod
    def _argmax_with_random_tie_breaking(preferences):
        """Compute probabilities greedily with respect to a set of preferences."""
        optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
        return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)

    @staticmethod
    def legal_epsilon_greedy(epsilon=None):
        """An epsilon-greedy distribution with illegal probabilities set to zero"""

        def sample_fn(key: rlax.ArrayLike,
                      preferences: rlax.ArrayLike,
                      legal: rlax.ArrayLike,
                      epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return DQNPolicy._categorical_sample(key, probs)

        def probs_fn(preferences: rlax.ArrayLike, legal: rlax.ArrayLike, epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            return DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)

        def logprob_fn(sample: rlax.ArrayLike,
                       preferences: rlax.ArrayLike,
                       legal: rlax.ArrayLike,
                       epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return rlax.base.batched_index(jnp.log(probs), sample)

        def entropy_fn(preferences: rlax.ArrayLike, legal: rlax.ArrayLike, epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return -jnp.nansum(probs * jnp.log(probs), axis=-1)

        return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)


    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def policy(network,
               net_params,
               epsilon: float,
               key: float,
               obs: rlax.ArrayLike,
               lm: rlax.ArrayLike):
        """Sample action from epsilon-greedy policy.

        Args:
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        # compute q
        q_vals = network.apply(net_params, obs)
        # set q for illegal actions to negative infinity
        q_vals = jnp.where(lm, q_vals, -jnp.inf)
        # compute actions
        actions = DQNPolicy.legal_epsilon_greedy(epsilon=epsilon).sample(key, q_vals, lm)
        return q_vals, actions

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def eval_policy(network, net_params, key, obs: rlax.ArrayLike, lm: rlax.ArrayLike):
        """Sample action from greedy policy.
        Args:
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        # compute q
        q_vals = network.apply(net_params, obs)
        # add large negative values to illegal actions
        q_vals = jnp.where(lm, q_vals, -jnp.inf)
        # compute actions
        return rlax.greedy().sample(key, q_vals)

class DQNLearning:
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def update_q(network, optimizer, online_params, trg_params, opt_state,
                 obs_tm1, a_tm1, obs_t, lm_t, r_t, term_t, discount_t):
        """Update network weights wrt Q-learning loss.

        Args:
            network    -- haiku Transformed network.
            optimizer  -- optimizer.
            net_params -- parameters (weights) of the network.
            opt_state  -- state of the optimizer.
            q_tm1      -- q-value of state-action at time t-1.
            obs_tm1    -- observation at time t-1.
            a_tm1      -- action at time t-1.
            r_t        -- reward at time t.
            term_t     -- terminal state at time t?
        """

        def q_learning_loss(online_params, trg_params, obs_tm1, a_tm1, obs_t,
                             lm_t, r_t, term_t, discount_t):
            q_tm1 = network.apply(online_params, obs_tm1)
            q_t = network.apply(trg_params, obs_t)
            # set q values of illegal actions to a large negative number.
            #  q_t = jnp.where(lm_t, q_t, -1e3)
            # set q values to zero if the state is terminal, i.e.
            #  q_t = jnp.where(jnp.broadcast_to(term_t, q_t.shape), 0.0, q_t)
            td_error = jax.vmap(rlax.q_learning, in_axes=(0, 0, 0, None, 0))(
                q_tm1, a_tm1[:, 0], r_t[:, 0], discount_t, q_t)
            return rlax.l2_loss(td_error).mean()

        dloss_dtheta = jax.grad(q_learning_loss)(online_params, trg_params, obs_tm1, a_tm1, obs_t,
                                                 lm_t, r_t, term_t, discount_t)
        updates, opt_state_t = optimizer.update(dloss_dtheta, opt_state)
        online_params_t = optix.apply_updates(online_params, updates)
        return online_params_t, opt_state_t

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def update_q_double(network, optimizer, online_params, trg_params, opt_state,
                 obs_tm1, a_tm1, obs_t, lm_t, r_t, term_t, discount_t):
        """Update network weights wrt Q-learning loss.

        Args:
            network    -- haiku Transformed network.
            optimizer  -- optimizer.
            net_params -- parameters (weights) of the network.
            opt_state  -- state of the optimizer.
            q_tm1      -- q-value of state-action at time t-1.
            obs_tm1    -- observation at time t-1.
            a_tm1      -- action at time t-1.
            r_t        -- reward at time t.
            term_t     -- terminal state at time t?
        """

        def double_q_learning_loss(online_params, trg_params, obs_tm1, a_tm1, obs_t,
                             lm_t, r_t, term_t, discount_t):
            q_tm1 = network.apply(online_params, obs_tm1)
            q_t = network.apply(trg_params, obs_t)
            q_sel = network.apply(online_params, obs_t)
            # set q values of illegal actions to a larger negative number.
            #  q_sel = jnp.where(lm_t, q_sel, -1e2)
            #  q_t = jnp.where(lm_t, q_t, -1e2)
            # set q values to zero if the state is terminal, i.e.
            #  q_t = jnp.where(jnp.broadcast_to(term_t, q_t.shape), 0.0, q_t)
            td_error = jax.vmap(rlax.double_q_learning, in_axes=(0, 0, 0, None, 0, 0))(
                q_tm1, a_tm1[:, 0], r_t[:, 0], discount_t, q_t, q_sel)
            return rlax.l2_loss(td_error).mean()

        dloss_dtheta = jax.grad(double_q_learning_loss)(online_params, trg_params, obs_tm1, a_tm1, obs_t,
                                                  lm_t, r_t, term_t, discount_t)
        updates, opt_state_t = optimizer.update(dloss_dtheta, opt_state)
        online_params_t = optix.apply_updates(online_params, updates)
        return online_params_t, opt_state_t

class DQNAgent:
    def __init__(self, observation_len, num_actions, target_update_period=None, discount=None,
                 epsilon=lambda x: 0.1, learning_rate=0.001, layers=None, use_double_q=True):
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))

        # Build and initialize Q-network.
        self.layers = layers or (512,)
        self.network = build_network(self.layers, num_actions)
        #  sample_input = env.observation_spec()["observation"].generate_value()
        sample_input = jnp.zeros(observation_len)
        self.trg_params = self.network.init(next(self.rng), sample_input)
        self.online_params = self.trg_params#self.network.init(next(self.rng), sample_input)

        # Build and initialize optimizer.
        self.optimizer = optix.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.online_params)
        self.epsilon = epsilon
        self.train_steps = 0
        self.target_update_period = target_update_period or 500
        self.discount = discount or 0.99
        if use_double_q:
            self.update_q = DQNLearning.update_q_double
        else:
            self.update_q = DQNLearning.update_q

    def exploit(self, observation, legal_actions):
        actions = DQNPolicy.eval_policy(
            self.network, self.online_params, next(self.rng), observation, legal_actions)
        return actions

    def explore(self, observation, legal_actions):
        q_vals, actions = DQNPolicy.policy(
            self.network, self.online_params,
            self.epsilon(self.train_steps), next(self.rng),
            observation, legal_actions)
        return q_vals, actions

    def train(self, observations_tm1, actions_tm1,
                    observations_t, legal_moves_t, rewards_t,
                    terminal_t):
        """Train the agent.

        Args:
            observations_tm1 -- observations at t-1.
            actions_tm1      -- actions at t-1.
            observations_t   -- observations at t.
            legal_moves_t    -- actions at t.
            rewards_t        -- rewards at t.
            terminal_t       -- terminal state at t?
        """
        self.online_params, self.opt_state = self.update_q(
            self.network,
            self.optimizer,
            self.online_params,
            self.trg_params,
            self.opt_state,
            observations_tm1, actions_tm1, observations_t, legal_moves_t,
            rewards_t, terminal_t, self.discount)

        if self.train_steps % self.target_update_period == 0:
            self.trg_params = self.online_params

        self.train_steps += 1

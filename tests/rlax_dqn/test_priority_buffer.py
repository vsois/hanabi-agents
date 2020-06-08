import numpy as onp
from hanabi_agents.rlax_dqn.priority_buffer import PriorityBuffer

def test_ctor():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 8

    prio_buf = PriorityBuffer(obs_len, lm_len, reward_len, capacity)

    assert prio_buf._obs_tm1_buf.shape == (capacity, obs_len)
    assert prio_buf._obs_t_buf.shape == (capacity, obs_len)
    assert prio_buf._act_tm1_buf.shape == (capacity, 1)
    assert prio_buf._lms_t_buf.shape == (capacity, lm_len)
    assert prio_buf._rew_t_buf.shape == (capacity, 1)

    assert prio_buf.capacity == capacity
    assert prio_buf.size == 0
    assert prio_buf.oldest_entry == 0

    #  assert prio_buf.sum_tree.capacity == capacity
    #  assert prio_buf.sum_tree.n_leafs == 8
    #  assert prio_buf.sum_tree.leaf_offset == 7
    #  assert len(prio_buf.sum_tree.heap) == 15
    assert prio_buf.max_priority == 1.0
    #  assert prio_buf.oldest_entry == 0
    #  assert prio_buf.sum_tree.depth == 3

def test_add_transition():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 8

    prio_buf = PriorityBuffer(obs_len, lm_len, reward_len, capacity)

    trns_size = 4
    obs1 = onp.random.randint(0, 2, (trns_size, obs_len))
    obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    while onp.all(obs1 == obs2):
        obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    assert not onp.all(obs1 == obs2)
    acts = onp.random.randint(0, lm_len + 1, (trns_size, 1))
    rew = onp.random.random((trns_size, 1))
    lms = onp.random.randint(0, 2, (trns_size, lm_len))
    term = onp.random.randint(0, 2, (trns_size, 1))

    prio_buf.add_transitions(
        obs1,
        acts,
        rew,
        obs2,
        lms,
        term)

    assert prio_buf.size == trns_size
    assert prio_buf.oldest_entry == trns_size

    assert onp.all(prio_buf._obs_tm1_buf[:trns_size] == obs1)
    assert onp.all(prio_buf._act_tm1_buf[:trns_size] == acts)
    assert onp.all(prio_buf._rew_t_buf[:trns_size] == rew)
    assert onp.all(prio_buf._obs_t_buf[:trns_size] == obs2)
    assert onp.all(prio_buf._lms_t_buf[:trns_size] == lms)
    assert onp.all(prio_buf._terminal_t_buf[:trns_size] == term)

    #  assert prio_buf.sum_tree.oldest == trns_size
    #  assert not prio_buf.sum_tree.full

    #  tree_leaf_offset = prio_buf.sum_tree.leaf_offset
    #  assert tree_leaf_offset == prio_buf.sum_tree.size - prio_buf.sum_tree.n_leafs
    #  assert prio_buf.sum_tree.n_leafs == capacity
    #  assert onp.all(prio_buf.sum_tree.heap[tree_leaf_offset:tree_leaf_offset + trns_size] == onp.ones(trns_size))


def test_sample_batch():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 8

    prio_buf = PriorityBuffer(obs_len, lm_len, reward_len, capacity)

    trns_size = 8
    obs1 = onp.random.randint(0, 2, (trns_size, obs_len))
    obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    while onp.all(obs1 == obs2):
        obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    assert not onp.all(obs1 == obs2)
    acts = onp.random.randint(0, lm_len + 1, (trns_size, 1))
    rew = onp.random.random((trns_size, 1))
    lms = onp.random.randint(0, 2, (trns_size, lm_len))
    term = onp.random.randint(0, 2, (trns_size, 1))

    prio_buf.add_transitions(
        obs1,
        acts,
        rew,
        obs2,
        lms,
        term)

    prio_buf.sample_batch(4)

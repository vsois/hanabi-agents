import numpy as onp
from hanabi_agents.rlax_dqn.experience_buffer import ExperienceBuffer

def test_ctor():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 7

    exp_buf = ExperienceBuffer(obs_len, lm_len, reward_len, capacity)

    assert exp_buf._obs_tm1_buf.shape == (capacity, obs_len)
    assert exp_buf._obs_t_buf.shape == (capacity, obs_len)
    assert exp_buf._act_tm1_buf.shape == (capacity, 1)
    assert exp_buf._lms_t_buf.shape == (capacity, lm_len)
    assert exp_buf._rew_t_buf.shape == (capacity, 1)

    assert exp_buf.capacity == capacity
    assert exp_buf.size == 0
    assert exp_buf.cur_idx == 0

def test_add_transition():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 7

    exp_buf = ExperienceBuffer(obs_len, lm_len, reward_len, capacity)

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

    exp_buf.add_transitions(
        obs1,
        acts,
        rew,
        obs2,
        lms,
        term)

    assert exp_buf.size == trns_size
    assert exp_buf.cur_idx == trns_size

    assert onp.all(exp_buf._obs_tm1_buf[:trns_size] == obs1)
    assert onp.all(exp_buf._act_tm1_buf[:trns_size] == acts)
    assert onp.all(exp_buf._rew_t_buf[:trns_size] == rew)
    assert onp.all(exp_buf._obs_t_buf[:trns_size] == obs2)
    assert onp.all(exp_buf._lms_t_buf[:trns_size] == lms)
    assert onp.all(exp_buf._terminal_t_buf[:trns_size] == term)

def test_add_transition_fill_capacity():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 7

    exp_buf = ExperienceBuffer(obs_len, lm_len, reward_len, capacity)

    trns_size = capacity
    obs1 = onp.random.randint(0, 2, (trns_size, obs_len))
    obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    while onp.all(obs1 == obs2):
        obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    assert not onp.all(obs1 == obs2)
    acts = onp.random.randint(0, lm_len + 1, (trns_size, 1))
    rew = onp.random.random((trns_size, 1))
    lms = onp.random.randint(0, 2, (trns_size, lm_len))
    term = onp.random.randint(0, 2, (trns_size, 1))

    exp_buf.add_transitions(
        obs1,
        acts,
        rew,
        obs2,
        lms,
        term)

    assert exp_buf.size == capacity
    assert exp_buf.cur_idx == 0

    assert onp.all(exp_buf._obs_tm1_buf == obs1)
    assert onp.all(exp_buf._act_tm1_buf == acts)
    assert onp.all(exp_buf._rew_t_buf == rew)
    assert onp.all(exp_buf._obs_t_buf == obs2)
    assert onp.all(exp_buf._lms_t_buf == lms)
    assert onp.all(exp_buf._terminal_t_buf == term)

def test_add_transition_capacity_overflow():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 7

    exp_buf = ExperienceBuffer(obs_len, lm_len, reward_len, capacity)

    trns_size = capacity + 1
    obs1 = onp.random.randint(0, 2, (trns_size, obs_len))
    obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    while onp.all(obs1 == obs2):
        obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    assert not onp.all(obs1 == obs2)
    acts = onp.random.randint(0, lm_len + 1, (trns_size, 1))
    rew = onp.random.random((trns_size, 1))
    lms = onp.random.randint(0, 2, (trns_size, lm_len))
    term = onp.random.randint(0, 2, (trns_size, 1))

    exp_buf.add_transitions(
        obs1,
        acts,
        rew,
        obs2,
        lms,
        term)

    assert exp_buf.size == capacity
    assert exp_buf.cur_idx == 1

    obs1[:1] = obs1[-1:]
    acts[:1] = acts[-1:]
    rew[:1] = rew[-1:]
    obs2[:1] = obs2[-1:]
    lms[:1] = lms[-1:]
    term[:1] = term[-1:]

    assert onp.all(exp_buf._obs_tm1_buf == obs1[:-1])
    assert onp.all(exp_buf._act_tm1_buf == acts[:-1])
    assert onp.all(exp_buf._rew_t_buf == rew[:-1])
    assert onp.all(exp_buf._obs_t_buf == obs2[:-1])
    assert onp.all(exp_buf._lms_t_buf == lms[:-1])
    assert onp.all(exp_buf._terminal_t_buf == term[:-1])

def test_getter():

    obs_len = 5
    lm_len = 3
    reward_len = 1
    capacity = 7

    exp_buf = ExperienceBuffer(obs_len, lm_len, reward_len, capacity)

    trns_size = capacity
    obs1 = onp.random.randint(0, 2, (trns_size, obs_len))
    obs2 = onp.random.randint(0, 2, (trns_size, obs_len))
    acts = onp.random.randint(0, lm_len + 1, (trns_size, 1))
    rew = onp.random.random((trns_size, 1))
    lms = onp.random.randint(0, 2, (trns_size, lm_len))
    term = onp.random.randint(0, 2, (trns_size, 1))

    exp_buf.add_transitions(
        obs1,
        acts,
        rew,
        obs2,
        lms,
        term)

    indices = [1, 3, 6]

    samples = exp_buf[indices]
    assert onp.all(samples.observation_tm1 == obs1[indices])
    assert onp.all(samples.action_tm1 == acts[indices])
    assert onp.all(samples.reward_t == rew[indices])
    assert onp.all(samples.observation_t == obs2[indices])
    assert onp.all(samples.legal_moves_t == lms[indices])
    assert onp.all(samples.terminal_t == term[indices])

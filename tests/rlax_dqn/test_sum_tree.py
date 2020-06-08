from hanabi_agents.rlax_dqn.sum_tree import SumTree
import numpy as np

def test_ctor():

    capacity = int(2**3)
    stree = SumTree(capacity)
    assert stree.capacity == capacity
    assert stree.n_leafs == 8
    assert stree.leaf_offset == 7
    assert len(stree.heap) == 15
    assert stree.max_priority == 1.0
    assert stree.oldest == 0
    assert not stree.full
    assert stree.depth == 3

def test_fill():

    capacity = int(2**3)
    stree = SumTree(capacity)
    for _ in range(capacity):
        stree.add_new()

    assert stree.oldest == 0
    assert stree.full
    assert np.all(stree.heap == np.array([8, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]))

def test_setter():

    capacity = int(2**3)
    stree = SumTree(capacity)
    for i in range(capacity):
        stree[i] = i

    assert stree.oldest == 0
    assert not stree.full
    assert stree.max_priority == capacity - 1
    assert np.all(stree.heap == np.array([28, 6, 22, 1, 5, 9, 13, 0, 1, 2, 3, 4, 5, 6, 7]))

def test_getter():

    capacity = int(2**3)
    stree = SumTree(capacity)
    for i in range(capacity):
        stree[i] = i * 2

    for i in range(capacity):
        assert stree[i] == i * 2

def test_leaf_index_to_heap():

    capacity = int(2**3)
    stree = SumTree(capacity)

    for i in range(capacity):
        assert stree._leaf_index_to_heap(i) == 7 + i

def test_path_to_root():

    capacity = int(2**3)
    stree = SumTree(capacity)
    
    path_to_root = [7, 3, 1, 0]

    for i in range(capacity):
        assert stree._path_to_root(i) == path_to_root
        path_to_root[0] += 1
        if path_to_root[0] % 2 == 1:
            path_to_root[1] += 1
            if path_to_root[1] % 2 == 1:
                path_to_root[2] += 1

def test_sample():

    capacity = int(2**3)
    stree = SumTree(capacity)
    for _ in range(capacity):
        stree.add_new()

    samples = [0, 0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(capacity + 1):
        assert stree.sample((i) / capacity) == samples[i]

    for i in range(capacity):
        stree[i] = i

    samples = [0, 3, 4, 5, 5, 6, 6, 7, 7]
    for i in range(capacity + 1):
        assert stree.sample((i) / capacity) == samples[i]

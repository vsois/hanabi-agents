from typing import NamedTuple, List, Callable, Union
import gin

@gin.configurable
class RlaxRainbowParams(NamedTuple):
    """Parameterization class for Rlax-based Rainbow DQN agent."""

    train_batch_size: int = 256
    target_update_period: int = 500
    discount: float = 0.99
    epsilon: Union[Callable[[int], float], float] = lambda x: 0.1
    learning_rate: float = 0.001
    layers: List[int] = [512]
    use_double_q: bool = True
    use_priority: bool = True
    experience_buffer_size: int = 2**19
    seed: int = 1234
    n_atoms: int = 51
    atom_vmax: int = 25
    beta_is: Union[Callable[[int], float], float] = lambda x: 0.4

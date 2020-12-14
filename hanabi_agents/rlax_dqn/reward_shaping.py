# imports
from .params import RewardShapingParams
import numpy as np
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from collections import Counter

# reward shaping class
class RewardShaper:
    
    def __init__(self,
                 params: RewardShapingParams = RewardShapingParams()):
        
        self.params = params

    def shape(self, obs1, obs2):
        [self._calculate_difference(o1, o2) for o1, o2 in zip(obs1, obs2)]
        return [self._calculate_difference(o1, o2) for o1, o2 in zip(obs1, obs2)]
    
    def level(self, obs):
        return [self._calculate_level(o) for o in obs]
    
    def _calculate_difference(self, obs1, obs2):
        return self._calculate_level(obs2) - self._calculate_level(obs1)
    
    def _calculate_level(self, obs):
        
        lt = obs.life_tokens
        it = obs.information_tokens
        fw = np.sum(obs.fireworks)
        
        max = obs.parent_game.max_score
        max_rank = obs.parent_game.num_ranks
        num_cards_per_rank = [obs.parent_game.number_card_instances(0,r) for r in range(obs.parent_game.num_ranks)]
        
        fireworks = obs.fireworks
        discard_counter = Counter(obs.discard_pile)
        
        for color, rank in enumerate(fireworks):
            
            for r in range(rank, max_rank):
                card = pyhanabi.HanabiCard(color, r)
                
                if discard_counter[card] == num_cards_per_rank[r]:
                    max -= (max_rank-r)
                    break
                     
        return lt * self.params.w_life_tokens + \
            it * self.params.w_info_tokens + \
            max * self.params.w_max_score + \
            fw * self.params.w_fireworks
            
        
        
            
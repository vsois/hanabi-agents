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
        return [self._calculate_difference(o1, o2) for o1, o2 in zip(obs1, obs2)]
    
    def level(self, obs):
        return zip(*[self._calculate_level(o) for o in obs])

    def _calculate_difference(self, obs1, obs2):
        return self._calculate_level(obs2)[0] - self._calculate_level(obs1)[0]
    
    def _calculate_level(self, obs):
        
        # aux vars
        fireworks = obs.fireworks
        discard_counter = Counter(obs.discard_pile)
        num_rank = obs.parent_game.num_ranks
        num_color = obs.parent_game.num_colors
        num_cards_per_rank = [obs.parent_game.number_card_instances(0,r) 
                              for r in range(obs.parent_game.num_ranks)]
        
        lt = obs.life_tokens
        it = obs.information_tokens
        fw = np.sum(fireworks) if lt > 0 else 0
        
        # max score
        max = obs.parent_game.max_score
        for color, rank in enumerate(fireworks):
            
            for r in range(rank, num_rank):
                card = pyhanabi.HanabiCard(color, r)
                
                if discard_counter[card] == num_cards_per_rank[r]:
                    max -= (num_rank-r)
                    break
        
        # calculate card knowledge index        
        playability_avg = obs.average_playability()
        # calculate common playability of players cards
        playability_common = np.array(obs.common_playability())
        # card knowledge
        card_knowledge = abs(playability_avg - playability_common)
        cki = np.sum(card_knowledge)
        
        level = lt * self.params.w_life_tokens + \
            it * self.params.w_info_tokens + \
            max * self.params.w_max_score + \
            fw * self.params.w_fireworks + \
            cki * self.params.w_card_knowledge
                     
        #print({"FW": fw, "IT": it, "LT": lt, "CKI": cki, "MS": max})
        return (level, {"FW": fw, "IT": it, "LT": lt, "CKI": cki, "MS": max})
        
        
            
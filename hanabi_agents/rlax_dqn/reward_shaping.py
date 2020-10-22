# imports
from .params import RewardShapingParams
import numpy as np
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from pyflakes.checker import counter

# reward shaping class
class RewardShaper:
    
    def __init__(self,
                 params: RewardShapingParams = RewardShapingParams()):
        
        self.params = params
        
#         self.counter_keys =('play', 'conservative')
#         self.counter = dict.fromkeys(counter_keys, 0)
        
    def shape(self, observations, moves):
        
        assert len(observations) == len(moves)
        
        # reset counter values
#         for key in self.counter:
#             self.counter[key] = 0
        
        return [self._calculate(obs, move)for obs, move in zip(observations, moves)]
        
    @property
    def conservative_plays(self):
            return "{}/{}".format(self.counter['conservative'],
                                  self.counter['play'])
                    
    def _calculate(self, observation, move):
                
        if move.move_type == pyhanabi.HanabiMove.Type.kPlay:
            return self._play_shape(observation, move)
        if move.move_type == pyhanabi.HanabiMove.Type.kDiscard:
            return self._discard_shape(observation, move)
        if move.move_type in [pyhanabi.HanabiMove.Type.kRevealColor,
                              pyhanabi.HanabiMove.Type.kRevealRank]:
            return self._hint_shape(observation, move)
            
    def _discard_shape(self, observation, move):
        return 0
    
    def _hint_shape(self, observation, move):
        return 0
    
    def _play_shape(self, observation, move):
        
        # the move may be illegal, eg. playing a card that is not available in hand
        try:
            prob = observation.playable_percent()[move.card_index]
        except IndexError:
            return 0
        
        if prob < self.params.min_play_probability:
            return self.params.w_play_probability

        return 0
    
        
        
            
            
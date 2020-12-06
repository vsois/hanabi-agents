# imports
from .params import RewardShapingParams
import numpy as np
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi

# reward shaping class
class RewardShaper:
    
    def __init__(self,
                 params: RewardShapingParams = RewardShapingParams()):
        
        self.params = params
        self.num_ranks = None

    def shape(self, observations, moves):
        
        assert len(observations) == len(moves)

        if self.num_ranks == None:
            counter = 0
            for obs in observations:
                self.num_ranks = obs.parent_game.num_ranks

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

        discard_pile = observation.discard_pile
        card_index = move.card_index
        discarded_card = observation.card_to_discard(card_index)
        
        if discarded_card.rank == self.num_ranks -1:
            return self.params.penalty_last_of_kind
        elif len(discard_pile) == 0:
            return 0
        elif discarded_card.rank > 0:
            for elem in discard_pile:
                if discarded_card.rank == elem.rank & discarded_card.color == elem.color:
                    return self.params.penalty_last_of_kind
            return 0
        else:
            counter = 0
            for elem in discard_pile:
                if elem.rank == 0 & elem.color == discarded_card.color:
                    counter += 1
            if counter == 2:
                return self.params.penalty_last_of_kind
            else:
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

    def get_params(self):
        return (self.params.w_play_probability, self.params.penalty_last_of_kind)

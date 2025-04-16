from pypokerengine.players import BasePokerPlayer
from src.bots.process_state_for_nn import process_poker_state_for_nn
import random

class RandomBot(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        # print(valid_actions)
        action_info = valid_actions[random.randint(0, len(valid_actions) - 1)]
        action, amount = action_info["action"], action_info["amount"]
        uuid = self.uuid

        capital = 0
        for player in round_state['seats']:
            if player['uuid'] == uuid:
                capital = player['stack']
                break

        if action == 'raise':
            if capital < amount['min']:
                return 'fold', 0
            return action, min(capital,amount['min'])

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
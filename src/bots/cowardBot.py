from pypokerengine.players import BasePokerPlayer

class CowardBot(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
        return action, amount   # action returned here is sent to the poker engine

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
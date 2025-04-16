from pypokerengine.players import BasePokerPlayer
import pypokerengine.utils.visualize_utils as U

class AllInBot(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        # print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
        #
        # print(f'Round state: \n{round_state}')
        # print(f'Valid actions: {valid_actions}')
        uuid = self.uuid
        capital = 0
        other_players_capital = []

        for player in round_state['seats']:
            other_players_capital.append(player['stack'])
            # print(f'PLAYER: {self.uuid}, {player['stack']}')
            if player['uuid'] == uuid:
                capital = player['stack']

        # print(f"OTHER CAPITAL: {other_players_capital}")
        for action in valid_actions:
            if action['action'] == 'raise':
                # print("We were just here!")
                other_player_min = min(other_players_capital)
                return 'raise', min(action['amount']['max'], capital, other_player_min)
        # Otherwise, just call.
        call_action_info = valid_actions[1]
        return call_action_info["action"], call_action_info["amount"]

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
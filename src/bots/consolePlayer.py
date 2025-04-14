import pypokerengine.utils.visualize_utils as U
from pypokerengine.players import BasePokerPlayer


class ConsolePlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
        action, amount = self._receive_action_from_console(valid_actions)
        return action, amount

    def receive_game_start_message(self, game_info):
        print(U.visualize_game_start(game_info, self.uuid))
        self._wait_until_input()

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))
        self._wait_until_input()

    def receive_street_start_message(self, street, round_state):
        print(U.visualize_street_start(street, round_state, self.uuid))
        self._wait_until_input()

    def receive_game_update_message(self, new_action, round_state):
        print(U.visualize_game_update(new_action, round_state, self.uuid))
        self._wait_until_input()

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
        self._wait_until_input()

    def _wait_until_input(self):
        input("Enter some key to continue ...")

    # FIXME: This code would be crash if receives invalid input.
    #        So you should add error handling properly.
    def _receive_action_from_console(self, valid_actions):
        # Convert valid_actions to a more accessible format
        actions_dict = {action['action']: action for action in valid_actions}

        # Display available actions to the player
        available_actions = list(actions_dict.keys())
        print(f"Available actions: {available_actions}")

        while True:
            try:
                # Get action input
                action = input("Enter action to declare (fold/call/raise) >> ")

                # Validate the action
                if action not in available_actions:
                    print(f"Invalid action. Please choose from {available_actions}")
                    continue

                # Handle each action type
                if action == 'fold':
                    amount = 0
                    break

                elif action == 'call':
                    amount = actions_dict['call']['amount']
                    break

                elif action == 'raise':
                    # Get and validate raise amount
                    min_amount = actions_dict['raise']['amount']['min']
                    max_amount = actions_dict['raise']['amount']['max']

                    print(f"Raise amount must be between {min_amount} and {max_amount}")
                    amount = input(f"Enter raise amount >> ")

                    try:
                        amount = int(amount)
                        if amount < min_amount or amount > max_amount:
                            print(f"Amount must be between {min_amount} and {max_amount}")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number")
                        continue

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again")

        return action, amount

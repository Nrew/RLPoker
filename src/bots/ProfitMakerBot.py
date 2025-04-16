# from pypokerengine.players import BasePokerPlayer
# import pypokerengine.utils.visualize_utils as U
#
# from ..model.model import PPO
# from ..model.wrapper import Agent
# from ..model.config import INITIAL_STACK, SMALL_BLIND, MAX_ROUND
#
# def train_use_ppo_bot():
#     ppo_algo = PPO()
#     ppo_agent = Agent(ppo_algo, player_name="ProfitMakerBot")
#     train_bot(ppo_algo, num_games = 10_000)
#     ppo_algo.save_model("ProfitMakerBot")
#     return ppo_agent
#
# def train_bot(ppo_algo, num_games):
#
#
# class AllInBot(BasePokerPlayer):
#
#     def declare_action(self, valid_actions, hole_card, round_state):
#
#
#     def receive_game_start_message(self, game_info):
#         pass
#
#     def receive_round_start_message(self, round_count, hole_card, seats):
#         pass
#
#     def receive_street_start_message(self, street, round_state):
#         pass
#
#     def receive_game_update_message(self, action, round_state):
#         pass
#
#     def receive_round_result_message(self, winners, hand_info, round_state):
#         pass
from pypokerengine.api.game import setup_config, start_poker
from bots.allInBot import AllInBot
from bots.basicBot import BasicBot

config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=AllInBot())
config.register_player(name="p2", algorithm=BasicBot())
config.register_player(name="p3", algorithm=BasicBot())
game_result = start_poker(config, verbose=1)
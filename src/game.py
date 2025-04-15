from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards
from bots.allInBot import AllInBot
from bots.randomBot import RandomBot
from bots.consolePlayer import ConsolePlayer
from bots.cowardBot import CowardBot

config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
config.register_player(name="All In Bot", algorithm=AllInBot())
config.register_player(name="RandomBot", algorithm=RandomBot())
config.register_player(name="Coward", algorithm=CowardBot())
# config.register_player(name="Console Player", algorithm=ConsolePlayer())
game_result = start_poker(config, verbose=2)
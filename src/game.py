from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards
from bots.allInBot import AllInBot
from bots.basicBot import BasicBot
from bots.consolePlayer import ConsolePlayer

config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
config.register_player(name="All In Bot", algorithm=AllInBot())
config.register_player(name="BasicBot", algorithm=BasicBot())
config.register_player(name="Console Player", algorithm=ConsolePlayer())
game_result = start_poker(config, verbose=2)
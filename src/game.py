from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards
from bots.allInBot import AllInBot
from bots.randomBot import RandomBot
from bots.consolePlayer import ConsolePlayer
from bots.cowardBot import CowardBot
import multiprocessing as mp
from multiprocessing import Manager
import time
from model.model import PPO
from model.wrapper import Agent
from model.config import MODEL_SAVE_PATH

def create_ppo_agent():

    ppo_algo = PPO()
    try:
        ppo_algo.load_latest(f"{MODEL_SAVE_PATH}")
        print(f"Successfully loaded PPO model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will use untrained model")

    ppo_agent = Agent(ppo_algo, player_name="ProfitMakerBot")

    return ppo_agent

def run_game(arg):
    i, counters = arg
    # Configure the game
    ppo_agent = create_ppo_agent()
    config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=50)

    config.register_player(name=f"RandomBot {i}", algorithm=RandomBot())
    config.register_player(name=f"AllInBot {i+1}", algorithm=AllInBot())

    config.register_player(name=f"RandomBot {i+2}", algorithm=RandomBot())

    config.register_player(name=f"RandomBot {i+3}", algorithm=RandomBot())
    config.register_player(name=f"RandomBot {i}", algorithm=RandomBot())
    config.register_player(name=f"ProfitMakerBot", algorithm=ppo_agent)


# config.register_player(name="Console Player", algorithm=ConsolePlayer())
    result = start_poker(config, verbose=0)

    for player in result['players']:
        if player['stack'] > 0:  # This player won
            bot_type = player['name'].split()[0]
            counters[bot_type] += 1
            break

    if i % 1000 == 0:
        print(f"Completed {i} games")

    return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    num_games = 10_000
    start_time = time.time()
    print(f"Initial start time: {start_time}")
    num_cores = mp.cpu_count()
    print(f"Running on {num_cores} CPU cores.")

    with Manager() as manager:
        counters = manager.dict({
            'AllInBot': 0,
            'RandomBot': 0,
            'CowardBot': 0,
            'ProfitMakerBot': 0
        })

        args = [(i, counters) for i in range(num_games)]

        with mp.Pool(processes=num_cores) as pool:
            pool.map(run_game, args)
        final_counts = dict(counters)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    print("\nWin distribution:")
    for bot_name, wins in final_counts.items():
        win_percentage = (int(wins) / num_games) * 100
        print(f"{bot_name}: {wins} wins ({win_percentage:.2f}%)")
import torch
import torch.multiprocessing as mp
import numpy as np
import os
import time
from pypokerengine.api.game import setup_config, start_poker
import random

from model import PPO
from wrapper import Agent
from bots.randomBot import RandomBot
from bots.allInBot import AllInBot
from bots.cowardBot import CowardBot
from memory import MemoryBuffer
import config


def run_training_episode(process_id, shared_dict, config_dict):
    """Function to run in parallel processes - each worker collects experiences"""
    ppo_algo = PPO()
    ppo_player_name = f"{config_dict['PLAYER_NAME_PREFIX']}_{process_id}"
    try:
        ppo_algo.load_latest(f"{config.MODEL_SAVE_PATH}")
        print(f"Worker {process_id} loaded latest model")
    except Exception as e:
        print(f"Worker {process_id} couldn't load latest model: {e}")
    ppo_player = Agent(ppo_algo, player_name=ppo_player_name)

    all_experiences = []
    current_game_states = []
    current_game_actions = []
    current_game_log_probs = []
    current_game_values = []
    current_game_final_reward = None

    original_select_action = ppo_algo.select_action
    def capturing_select_action(state_np):
        action_idx, log_prob, value = original_select_action(state_np)
        if state_np is not None:
            current_game_states.append(state_np.copy() if hasattr(state_np, 'copy') else state_np)
            current_game_actions.append(action_idx)
            current_game_log_probs.append(log_prob.item() if hasattr(log_prob, 'item') else log_prob)
            current_game_values.append(value.item() if hasattr(value, 'item') else value)
        return action_idx, log_prob, value
    ppo_algo.select_action = capturing_select_action

    original_finish_hand = ppo_algo.finish_hand
    def capturing_finish_hand(final_reward):
        nonlocal current_game_final_reward
        current_game_final_reward = final_reward
        return original_finish_hand(final_reward)
    ppo_algo.finish_hand = capturing_finish_hand

    game_config = setup_config(
        max_round=config_dict['MAX_ROUND'],
        initial_stack=config_dict['INITIAL_STACK'],
        small_blind_amount=config_dict['SMALL_BLIND'],
        ante=config_dict['ANTE']
    )

    # Make the game with random players and bots for wider variety.
    slots_remaining = 6
    game_config.register_player(name=ppo_player.player_name, algorithm=ppo_player)
    slots_remaining -= 1

    additional_ppo = random.randint(0, min(2, slots_remaining))  # Max 2 more (3 total)
    for i in range(additional_ppo):
        game_config.register_player(name=f"PPO_{process_id}_{i}", algorithm=ppo_player)
        slots_remaining -= 1

    num_allin = random.randint(0, min(2, slots_remaining))
    for i in range(num_allin):
        game_config.register_player(name=f"AllInBot_{i}", algorithm=AllInBot())
        slots_remaining -= 1

    num_coward = random.randint(0, min(3, slots_remaining))
    for i in range(num_coward):
        game_config.register_player(name=f"CowardBot_{i}", algorithm=CowardBot())
        slots_remaining -= 1

    for i in range(slots_remaining):
        game_config.register_player(name=f"RandomBot_{process_id}_{i}", algorithm=RandomBot())

    games_played = 0
    while games_played < config_dict['games_per_worker']:
        try:
            current_game_states = []
            current_game_actions = []
            current_game_log_probs = []
            current_game_values = []
            current_game_final_reward = None

            game_result = start_poker(game_config, verbose=0)
            games_played += 1

            ppo_final_stack = config_dict['INITIAL_STACK']
            for p in game_result.get('players', []):
                if p.get('name') == ppo_player.player_name:
                    ppo_final_stack = p.get('stack', config_dict['INITIAL_STACK'])
                    break

            final_reward = ppo_final_stack - config_dict['INITIAL_STACK']

            final_reward_to_use = current_game_final_reward
            if final_reward_to_use is None:
                final_reward_to_use = final_reward
            if len(current_game_states) > 0:
                experience = {
                    'states': current_game_states.copy(),
                    'actions': current_game_actions.copy(),
                    'log_probs': current_game_log_probs.copy(),
                    'values': current_game_values.copy(),
                    'rewards': [0] * (len(current_game_states) - 1) + [final_reward_to_use],
                    'dones': [False] * (len(current_game_states) - 1) + [True]
                }
                all_experiences.append(experience)
            else:
                print(f"Worker {process_id} collected no experiences in game {games_played}")

        except Exception as e:
            print(f"Error in worker {process_id}, game {games_played}: {e}")
            import traceback
            traceback.print_exc()

    total_experiences = sum(len(exp['states']) for exp in all_experiences)
    with shared_dict['lock']:
        for exp in all_experiences:
            shared_dict['experiences'].append(exp)
        shared_dict['games_completed'] += games_played

    return

def main():
    device = config.DEVICE
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    os.makedirs("checkpoints", exist_ok=True)

    ppo_algo = PPO(device=device)
    if config.LOAD_MODEL:
        ppo_algo.load_model(config.MODEL_SAVE_PATH)

    config_dict = {
        'MAX_ROUND': config.MAX_ROUND,
        'INITIAL_STACK': config.INITIAL_STACK,
        'SMALL_BLIND': config.SMALL_BLIND,
        'ANTE': config.ANTE,
        'PLAYER_NAME_PREFIX': config.PLAYER_NAME_PREFIX,
        'NUM_OPPONENTS': config.NUM_OPPONENTS,
        'OPPONENT_TYPE': config.OPPONENT_TYPE,
        'BATCH_SIZE': config.BATCH_SIZE,            # Add these for memory buffer in workers
        'BUFFER_SIZE': config.BUFFER_SIZE,
        'games_per_worker': config.GAMES_PER_WORKER 
    }

    num_cores = mp.cpu_count()
    print(f"Running with {num_cores} processes")

    start_time = time.time()
    total_games_played = 0
    updates_done = 0
    next_save_threshold = 500
    num_iterations = config.NUM_TRAINING_GAMES // (config_dict['games_per_worker'] * num_cores) + 1

    for training_iteration in range(1, num_iterations + 1):
        manager = mp.Manager()
        shared_dict = manager.dict({
            'experiences': manager.list(),
            'games_completed': 0,
            'lock': manager.Lock()
        })
        processes = []
        for i in range(num_cores):
            p = mp.Process(target=run_training_episode, args=(i, shared_dict, config_dict))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        with shared_dict['lock']:
            experiences = list(shared_dict['experiences'])
            games_completed = shared_dict['games_completed']

        total_games_played += games_completed
        print(f"Iteration {training_iteration}: Collected data from {games_completed} games")

        valid_experiences = [exp for exp in experiences if len(exp.get('states', [])) > 0]
        print(f"Processing {len(valid_experiences)} valid experience batches")

        for exp in valid_experiences:
            for i in range(len(exp['states'])):
                ppo_algo.memory.store_step(
                    exp['states'][i],
                    exp['actions'][i],
                    exp['log_probs'][i],
                    exp['values'][i]
                )
            final_reward = exp['rewards'][-1] if exp['rewards'] else 0
            ppo_algo.memory.finish_trajectory(final_reward)

        print(f"Memory buffer: {len(ppo_algo.memory)} samples, Ready: {ppo_algo.memory.ready()}")

        if ppo_algo.memory.ready() or ppo_algo.memory.is_full():
            print(f"Updating with {len(ppo_algo.memory)} samples")
            update_start = time.time()
            update_success = ppo_algo.update()
            update_time = time.time() - update_start

            if update_success:
                updates_done += 1
                print(f"Update {updates_done} completed in {update_time:.2f}s")

                try:
                    ppo_algo.save_model(f"{config.MODEL_SAVE_PATH}", "")
                    print("Saved latest model for workers")
                except Exception as e:
                    print(f"Error saving latest model: {e}")

        if total_games_played >= next_save_threshold:
            ppo_algo.save_model(config.MODEL_SAVE_PATH, f"{total_games_played}")
            next_save_threshold += config.SAVE_INTERVAL
            elapsed = time.time() - start_time
            print(f"\nProgress: {total_games_played}/{config.NUM_TRAINING_GAMES} games, {updates_done} updates")
            print(f"Time elapsed: {elapsed:.2f}s")

    # Final save
    ppo_algo.save_model(config.MODEL_SAVE_PATH, "final")
    print(f"\nTraining complete: {total_games_played} games, {updates_done} updates")
    print(f"Total time: {(time.time() - start_time):.2f}s")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
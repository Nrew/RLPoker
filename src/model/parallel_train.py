import torch
import torch.multiprocessing as mp
import numpy as np
import os
import time
from pypokerengine.api.game import setup_config, start_poker

# Import your modules
from model import PPO
from wrapper import Agent
from bots.randomBot import RandomBot
from bots.allInBot import AllInBot
import config

def run_training_episode(process_id, shared_dict, config_dict):
    """Function to run in parallel processes"""
    # Force CPU for worker processes to avoid CUDA issues
    device = torch.device("cpu")
    ppo_algo = PPO(device=device)  # Use CPU for workers
    ppo_player_name = f"{config_dict['PLAYER_NAME_PREFIX']}_{process_id}"
    ppo_player = Agent(ppo_algo, player_name=ppo_player_name)

    # Add direct experience collection
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

    game_config.register_player(name=ppo_player.player_name, algorithm=ppo_player)
    game_config.register_player(name=f"{ppo_player.player_name}_{process_id}", algorithm=ppo_player)
    game_config.register_player(name="RandomBot", algorithm=RandomBot())
    game_config.register_player(name="RandomBot", algorithm=RandomBot())
    game_config.register_player(name="RandomBot", algorithm=RandomBot())
    game_config.register_player(name="AllInBot", algorithm=AllInBot())

    # Run games in a loop
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

            player_results = []

            for player_name in [ppo_player.player_name, f"{ppo_player.player_name}_{process_id}"]:
                ppo_final_stack = config_dict['INITIAL_STACK']
                for p in game_result.get('players', []):
                    if p.get('name') == player_name:
                        ppo_final_stack = p.get('stack', config_dict['INITIAL_STACK'])
                        break

                final_reward = ppo_final_stack - config_dict['INITIAL_STACK']
                player_results.append(final_reward)

            final_reward_to_use = current_game_final_reward
            if final_reward_to_use is None:
                final_reward_to_use = np.mean(player_results)

            if len(current_game_states) > 0:
                # Create experience record
                experience = {
                    'states': current_game_states.copy(),
                    'actions': current_game_actions.copy(),
                    'log_probs': current_game_log_probs.copy(),
                    'values': current_game_values.copy(),
                    'rewards': [0] * (len(current_game_states) - 1) + [final_reward_to_use] if len(current_game_states) > 0 else [],
                    'dones': [False] * (len(current_game_states) - 1) + [True] if len(current_game_states) > 0 else []
                }
                all_experiences.append(experience)

                print(f"Worker {process_id} collected {len(current_game_states)} steps in game {games_played}")
            else:
                print(f"Worker {process_id} collected no experiences in game {games_played}")

            if games_played % 10 == 0:
                print(f"Worker {process_id} completed {games_played} games")

        except Exception as e:
            print(f"Error in worker {process_id}, game {games_played}: {e}")
            import traceback
            traceback.print_exc()

    total_experiences = sum(len(exp['states']) for exp in all_experiences)
    with shared_dict['lock']:
        for exp in all_experiences:
            shared_dict['experiences'].append(exp)
        shared_dict['games_completed'] += games_played
        print(f"Worker {process_id} finished after playing {games_played} games with {total_experiences} total experience steps")

    return

def main():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    os.makedirs("../checkpoints", exist_ok=True)

    device = config.DEVICE
    ppo_algo = PPO(device=device)

    if config.LOAD_MODEL:
        ppo_algo.load_model(config.MODEL_SAVE_PATH)

    # Config dictionary to pass to processes
    config_dict = {
        'MAX_ROUND': config.MAX_ROUND,
        'INITIAL_STACK': config.INITIAL_STACK,
        'SMALL_BLIND': config.SMALL_BLIND,
        'ANTE': config.ANTE,
        'PLAYER_NAME_PREFIX': config.PLAYER_NAME_PREFIX,
        'NUM_OPPONENTS': config.NUM_OPPONENTS,
        'OPPONENT_TYPE': config.OPPONENT_TYPE,
        'games_per_worker': 20  # Each worker plays this many games before reporting back
    }

    # Setup for parallel execution
    num_cores = mp.cpu_count()
    print(f"Running with {num_cores} processes")

    # Training loop
    start_time = time.time()
    total_games_played = 0
    updates_done = 0

    num_iterations = config.NUM_TRAINING_GAMES // (config_dict['games_per_worker'] * num_cores) + 1

    for training_iteration in range(1, num_iterations + 1):
        # Create a manager for shared data
        manager = mp.Manager()
        shared_dict = manager.dict({
            'experiences': manager.list(),
            'games_completed': 0,
            'lock': manager.Lock()
        })

        # Create and start processes
        processes = []
        for i in range(num_cores):
            p = mp.Process(target=run_training_episode, args=(i, shared_dict, config_dict))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Process collected experiences
        with shared_dict['lock']:
            raw_data_list = list(shared_dict['experiences'])
            games_completed = shared_dict['games_completed']

            print(f"Collected raw data from {games_completed} games")

            print(f"Iteration {training_iteration}: Completed {games_completed} games, collected experiences")
            total_games_played += games_completed

        all_experiences = list(shared_dict['experiences'])
        total_exp_count = sum(1 for exp in all_experiences if len(exp.get('states', [])) > 0)

        print(f"Main process received {total_exp_count} valid experience batches")

        # Add experiences to PPO memory
        for exp in all_experiences:
            if len(exp.get('states', [])) > 0:
                for i in range(len(exp['states'])):
                    # Get data for this step
                    state = exp['states'][i]
                    action = exp['actions'][i]
                    log_prob = exp['log_probs'][i] if i < len(exp['log_probs']) else 0.0
                    value = exp['values'][i] if i < len(exp['values']) else 0.0
                    reward = exp['rewards'][i] if i < len(exp['rewards']) else 0.0
                    done = exp['dones'][i] if i < len(exp['dones']) else False

                    # Add directly to memory
                    ppo_algo.memory.states.append(state)
                    ppo_algo.memory.actions.append(action)
                    ppo_algo.memory.log_probs.append(log_prob)
                    ppo_algo.memory.values.append(value)
                    ppo_algo.memory.rewards.append(reward)
                    ppo_algo.memory.dones.append(done)
                    ppo_algo.memory._current_size += 1
        print(f"Main PPO memory buffer size: {len(ppo_algo.memory)}")
        print(f"Memory ready for update: {ppo_algo.memory.ready()}")
        print(f"Memory full: {ppo_algo.memory.is_full()}")

        # Now update if ready (or force if needed)
        if ppo_algo.memory.ready() or ppo_algo.memory.is_full():
            print(f"Updating with {len(ppo_algo.memory)} samples")
            update_start = time.time()
            update_performed = ppo_algo.update()
            update_time = time.time() - update_start

            if update_performed:
                updates_done += 1
                print(f"Update {updates_done} completed in {update_time:.2f}s")
            else:
                print("Update failed!")


        # Save model periodically
        if training_iteration % config.SAVE_INTERVAL == 0:
            try:
                ppo_algo.save_model(config.MODEL_SAVE_PATH, training_iteration)
                current_time = time.time()
                elapsed = current_time - start_time
                print(f"\nIteration: {training_iteration} | Games: {total_games_played} | Updates: {updates_done}")
                print(f"Time Elapsed: {elapsed:.2f}s")
            except Exception as e:
                print(f"Error saving model: {e}")

    # Final save
    ppo_algo.save_model(config.MODEL_SAVE_PATH, "final")
    print("\n--- Training Finished ---")
    print(f"Total games played: {total_games_played}")
    print(f"Total PPO updates: {updates_done}")
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)
    main()
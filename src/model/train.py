import time
import numpy as np
from pypokerengine.api.game import setup_config, start_poker

try:
    from . import config
    from .model import PPO
    from .wrapper import Agent
    from ..bots.randomBot import RandomBot
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import config
    from model import PPO
    from wrapper import Agent
    from bots.randomBot import RandomBot

def choose_opponent(opponent_type="call"):
    """Helper function to select an opponent player instance."""
    return RandomBot()

def main():
    """Main training loop."""
    print("Initializing PPO Algorithm and Player...")
    ppo_algo = PPO(device=config.DEVICE)
    ppo_player_name = f"{config.PLAYER_NAME_PREFIX}_0"
    ppo_player = Agent(ppo_algo, player_name=ppo_player_name)

    if config.LOAD_MODEL:
        ppo_algo.load_model(config.MODEL_SAVE_PATH)

    # --- Opponent Setup ---
    opponents = []
    opponent_names = []
    print(f"Setting up {config.NUM_OPPONENTS} opponents of type: {config.OPPONENT_TYPE}")
    for i in range(config.NUM_OPPONENTS):
        opponent_instance = choose_opponent(config.OPPONENT_TYPE)
        opponents.append(opponent_instance)
        opponent_names.append(f"Opponent_{i}") # Ensure unique names

    # --- Training Loop ---
    start_time = time.time()
    total_steps = 0
    games_played = 0
    updates_done = 0
    all_game_rewards = [] # Track final stack relative to start for rough progress


    print(f"Starting training for {config.NUM_TRAINING_GAMES} games...")
    for game_num in range(1, config.NUM_TRAINING_GAMES + 1):
        # print(f"\n--- Starting Game {game_num}/{config.NUM_TRAINING_GAMES} ---")

        # Setup game configuration for each new game
        game_config = setup_config(
            max_round=config.MAX_ROUND,
            initial_stack=config.INITIAL_STACK,
            small_blind_amount=config.SMALL_BLIND,
            ante=config.ANTE
        )

        # Register players - The engine assigns UUIDs based on registration order/names
        # Our player handles receiving its UUID via receive_game_start_message
        game_config.register_player(name=ppo_player.player_name, algorithm=ppo_player)

        # Register Opponent Players
        for name, player_obj in zip(opponent_names, opponents):
            game_config.register_player(name=name, algorithm=player_obj)

        # Run the game
        try:
            # verbose=0 suppresses PyPokerEngine's default console output
            game_result = start_poker(game_config, verbose=0)
            games_played += 1

            # Log game outcome
            ppo_final_stack = config.INITIAL_STACK # Default if not found
            for p in game_result.get('players', []):
                 # Use name matching as UUID might change if engine re-registers across games
                 if p.get('name') == ppo_player.player_name:
                    ppo_final_stack = p.get('stack', config.INITIAL_STACK)
                    break
            game_reward = ppo_final_stack - config.INITIAL_STACK
            all_game_rewards.append(game_reward)


            # --- PPO Update Check ---
            # Check if buffer has enough samples OR if buffer is full (force update)
            if ppo_algo.memory.ready() or ppo_algo.memory.is_full():
                update_performed = ppo_algo.update()
                if update_performed:
                    updates_done += 1

            # --- Logging and Saving ---
            if game_num % config.SAVE_INTERVAL == 0:
                ppo_algo.save_model(config.MODEL_SAVE_PATH, game_num)
                # Log progress
                current_time = time.time()
                elapsed = current_time - start_time
                avg_reward = np.mean(all_game_rewards[-config.SAVE_INTERVAL:]) # Avg reward over last interval
                buffer_fill = len(ppo_algo.memory) / config.BUFFER_SIZE * 100
                print(f"\nGame: {game_num}/{config.NUM_TRAINING_GAMES} | Updates: {updates_done}")
                print(f"Time Elapsed: {elapsed:.2f}s | Avg Reward (last {config.SAVE_INTERVAL} games): {avg_reward:.2f}")
                # print(f"Total Steps Processed (approx): {ppo_algo.learn_step_counter * config.BATCH_SIZE}") # Estimate


        except Exception as e:
             print(f"\n--- Error during Game {game_num} ---")
             print(e)
             import traceback
             traceback.print_exc()
             print("Skipping game result and potential update.")
             # Clear any potentially corrupted trajectory data from memory
             ppo_algo.memory._clear_trajectory_buffer()
             return # Move to the next game

    # --- Final Save ---
    ppo_algo.save_model(config.MODEL_SAVE_PATH, 'final')
    print("\n--- Training Finished ---")
    print(f"Total games played: {games_played}")
    print(f"Total PPO updates: {updates_done}")
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time):.2f} seconds")
    # Plot rewards if desired
    # import matplotlib.pyplot as plt
    # plt.plot(all_game_rewards)
    # plt.title("Game Rewards (Final Stack - Initial Stack)")
    # plt.xlabel("Game Number")
    # plt.ylabel("Reward")
    # plt.show()


if __name__ == "__main__":
    main()
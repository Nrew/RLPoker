# poker_env_wrapper.py
from pypokerengine.players import BasePokerPlayer
import numpy as np

from . import config
from .utils import extract_state, map_action_to_poker
from .model import PPO

class Agent(BasePokerPlayer):
    """
    PyPokerEngine Player that uses the PPOAlgorithm to make decisions
    and collects experience.
    """
    def __init__(self, ppo: PPO, player_name="PPO_RL_Agent"):
        super().__init__()
        self.algorithm = ppo
        self.player_name = player_name
        self.uuid = self.generate_uuid() # Initial temporary UUID

        # Hand-specific state
        self.current_stack = 0
        self.initial_stack_this_game = 0
        self.stack_at_round_start = 0
        self.last_hole_card = None
        self.last_round_state = None

    def set_uuid(self, uuid):
        """Called externally to set the official UUID from the engine."""
        self.uuid = uuid
        # print(f"Agent {self.player_name} assigned UUID: {self.uuid}")

    # --- Action Handling ---
    def declare_action(self, valid_actions, hole_card, round_state):
        self.last_hole_card = hole_card # Store for potential use later
        self.last_round_state = round_state # Store for state extraction/action mapping

        # 1. Extract State (NumPy array)
        state_np = extract_state(
            hole_card,
            round_state,
            self.uuid,
            self.initial_stack_this_game,
            valid_actions=valid_actions)

        if state_np is None:
            print(f"Warning: State extraction failed or returned wrong shape in declare_action. Shape: {state_np.shape if state_np is not None else 'None'}. Folding.")             # Find fold action if possible
            fold_action = next((a for a in valid_actions if a['action'] == 'fold'), {'action':'fold', 'amount':0})
            return fold_action['action'], fold_action['amount']

        # 2. Select Action using PPO policy (also stores step in memory buffer)
        action_idx, _, _ = self.algorithm.select_action(state_np) # log_prob, value stored internally

        # 3. Map agent's action index to a valid poker action
        action_name, amount = map_action_to_poker(
            action_idx,
            valid_actions,
            self.current_stack,
            round_state
        )

        # print(f"Agent {self.uuid} declaring: {action_name} {amount} (idx: {action_idx})") # Debug
        return action_name, amount

    # --- Game Lifecycle Callbacks ---
    def game_start(self, game_info):
        """Called once at the start of a game."""
        my_info = next((p for p in game_info['seats'] if p['uuid'] == self.uuid), None)
        if my_info:
             self.initial_stack_this_game = my_info['stack']
             self.current_stack = self.initial_stack_this_game
             # print(f"Game start for {self.uuid}. Initial stack: {self.initial_stack_this_game}")
        else:
             print(f"Warning: Could not find agent {self.uuid} in game_start game_info.")
             self.initial_stack_this_game = config.INITIAL_STACK # Fallback
             self.current_stack = self.initial_stack_this_game

    def round_start(self, round_count, hole_card, seats):
        """Called at the start of each hand (round)."""
        my_info = next((p for p in seats if p['uuid'] == self.uuid), None)
        if my_info:
            self.current_stack = my_info['stack'] # Update stack at round start
            self.stack_at_round_start = self.current_stack
        else:
             print(f"Warning: Could not find agent {self.uuid} in round_start seats.")
             # Keep previous stack? Or reset? Let's keep it.

        # Clear the temporary trajectory buffer in memory for the new hand
        # Only clear if the algorithm and memory exist
        if hasattr(self.algorithm, 'memory') and self.algorithm.memory:
             self.algorithm.memory._clear_trajectory_buffer()
        
        self.last_hole_card = hole_card # Store initial hole cards
        # print(f"--- Round {round_count} start for {self.uuid}, Stack: {self.current_stack} ---")


    def street_start(self, street, round_state):
        """Called at the start of each betting street."""
        # Update internal state if necessary, e.g., current stack
        my_info = next((p for p in round_state['seats'] if p['uuid'] == self.uuid), None)
        if my_info:
            self.current_stack = my_info['stack']
        self.last_round_state = round_state # Store state at street start


    def game_update(self, game_info):
         """Called frequently with game updates (e.g., after each action)."""
         # Update current stack for accurate action mapping (esp. all-in checks)
         my_info = next((p for p in game_info['seats'] if p['uuid'] == self.uuid), None)
         if my_info:
             self.current_stack = my_info['stack']


    def round_result(self, winners, hand_info, round_state):
        """Called at the end of a hand."""
        # Find the agent's final state in the round result
        my_info = next((p for p in round_state['seats'] if p['uuid'] == self.uuid), None)

        if my_info:
            final_stack = my_info['stack']
        else:
            # Agent not found in final seats, likely busted or error. Assume 0 stack.
            # print(f"Info: Agent {self.uuid} not found in round_result seats. Assuming final stack is 0.")
            final_stack = 0

        # Retrieve the stack stored at the beginning of this round
        start_stack = self.stack_at_round_start

        # Calculate the raw reward (change in stack size for this hand)
        raw_reward = final_stack - start_stack

        # --- Reward Normalization (Optional but Recommended) ---
        # Normalize by the Big Blind amount used in the config
        # Ensure Big Blind is non-zero
        bb = max(config.BIG_BLIND, config.EPSILON)
        normalized_reward = raw_reward / bb
        # --- Choose which reward to use ---
        reward_to_use = normalized_reward # Or use raw_reward if you prefer

        # Finalize the trajectory in the PPO memory buffer with the calculated reward
        # Ensure algorithm and finish_hand exist before calling
        if hasattr(self.algorithm, 'finish_hand'):
            self.algorithm.finish_hand(reward_to_use)
        else:
             print(f"Warning: PPO Algorithm or finish_hand method not found during round_result for {self.uuid}")


        # Optional: Log the result
        # print(f"Round result for {self.uuid}. Start: {start_stack}, Final: {final_stack}, Reward: {raw_reward} (Norm: {reward_to_use:.2f})")
        # print(f"Memory buffer size after hand: {len(self.algorithm.memory)}")

        # Reset potentially sensitive hand-specific info (optional, round_start should overwrite)
        self.last_hole_card = None
        self.last_round_state = None
        # Don't reset stack_at_round_start here, it's set by the *next* round_start

    def game_result(self, game_info):
        """Called at the end of the game."""
        my_info = next((p for p in game_info['seats'] if p['uuid'] == self.uuid), None)
        final_stack = my_info['stack'] if my_info else -1
        print(f"--- Game Over for {self.player_name} ({self.uuid}) --- Final Stack: {final_stack}")


    # --- Message Receivers (Map to internal methods) ---
    # Need these to get the agent's UUID correctly assigned by the engine

    def receive_game_start_message(self, game_info):
        # Find own UUID based on name registration
        my_info = next((p for p in game_info['seats'] if p['name'] == self.player_name), None)
        if my_info:
            self.set_uuid(my_info['uuid'])
        else:
            print(f"ERROR: Could not find player name '{self.player_name}' in game_start message!")
            # Assign a random one? Engine might fail.
            self.set_uuid(self.generate_uuid())
        self.game_start(game_info) # Call the internal handler


    def receive_round_start_message(self, round_count, hole_card, seats):
        # Store stack at the start of the round for reward calculation
        self.round_start(round_count, hole_card, seats)

    def receive_street_start_message(self, street, round_state):
        self.street_start(street, round_state)

    def receive_game_update_message(self, new_action, game_state):
        # Update internal state based on the latest game state
        self.game_update(game_state) # Pass the full game state

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.round_result(winners, hand_info, round_state)

    def receive_game_result_message(self, game_info):
        self.game_result(game_info)
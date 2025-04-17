# config.py
import torch

# --- PPO Hyperparameters ---
STATE_DIM = 30                  # Dimension of the state vector [Original (21) + Win Rate (1) + Pot Odds (1) + SPR (1) + Bet Count (1) + Board Texture (3) = 28] (update if features change) 
ACTION_DIM = 5                  # Discrete actions: Fold, Call, Raise Min, Raise Pot, All-in
GAMMA = 0.99                    # Discount factor for rewards
PPO_EPSILON = 0.2               # PPO clipping parameter (ratio constraint)
PPO_EPOCHS = 5                  # Number of optimization epochs per batch update
LEARNING_RATE_ACTOR = 1e-4      # Learning rate for the actor network
LEARNING_RATE_CRITIC = 3e-4     # Learning rate for the critic network (often slightly higher)
ENTROPY_BETA = 0.01             # Coefficient for the entropy bonus (encourages exploration)
GAE_LAMBDA = 0.95               # Lambda parameter for Generalized Advantage Estimation
VALUE_LOSS_COEFF = 0.5          # Coefficient for the critic's value loss

# --- Training Configuration ---
BATCH_SIZE = 256                # Number of steps (transitions) per PPO update batch
BUFFER_SIZE = BATCH_SIZE * 15   # Max steps to store in memory before forcing update
NUM_TRAINING_GAMES = 2_000      # Total number of games to simulate for training
INITIAL_STACK = 1000            # Starting stack size for players
SMALL_BLIND = 5                 # Small blind amount
BIG_BLIND = SMALL_BLIND * 2     # Big blind amount for normilazation
ANTE = 0                        # Ante amount
MAX_ROUND = 15                  # Max rounds per game (keeps games from running infinitely long)
MODEL_SAVE_PATH = "../checkpoints/poker_ppo"   # Prefix for saving model files
SAVE_INTERVAL = 500             # Save model every N games
LOAD_MODEL = False              # Set to True to load a pre-trained model at the start

# --- Environment/Agent Settings ---
NUM_OPPONENTS = 6               # Number of opponents in the game (including the agent)
OPPONENT_TYPE = "call"          # Opponent type ('call', 'fold', 'random', 'ppo' - requires another agent)
PLAYER_NAME_PREFIX = "Agent_"   # Prefix for naming player instances

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Network Architecture ---
HIDDEN_UNITS = 512              # Number of units in hidden layers

# --- Constants ---
MAX_PLAYERS = NUM_OPPONENTS + 1 # Total players in the game (including the agent)
EPSILON = 1e-6                  # Small value to prevent division by zero in calculations
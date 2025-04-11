# config.py
import torch

# --- PPO Hyperparameters ---
STATE_DIM = 20                  # Dimension of the state vector (update if features change)
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
BATCH_SIZE = 128                # Number of steps (transitions) per PPO update batch
BUFFER_SIZE = BATCH_SIZE * 10   # Max steps to store in memory before forcing update
NUM_TRAINING_GAMES = 10000      # Total number of games to simulate for training
INITIAL_STACK = 1000            # Starting stack size for players
SMALL_BLIND = 5                 # Small blind amount
ANTE = 0                        # Ante amount
MAX_ROUND = 10                  # Max rounds per game (keeps games from running infinitely long)
MODEL_SAVE_PATH = "poker_ppo"   # Prefix for saving model files
SAVE_INTERVAL = 200             # Save model every N games
LOAD_MODEL = False              # Set to True to load a pre-trained model at the start

# --- Environment/Agent Settings ---
OPPONENT_TYPE = "call"          # Opponent type ('call', 'fold', 'random', 'ppo' - requires another agent)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Network Architecture ---
HIDDEN_UNITS = 128     # Number of units in hidden layers
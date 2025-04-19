# Reinforcement Learning for Texas Hold'em Poker

A reinforcement learning implementation that applies Proximal Policy Optimization (PPO) to train an AI agent to play No-Limit Texas Hold'em Poker using the PyPokerEngine framework that can:

- Extract features from poker game states
- Make strategic decisions (fold, call, or raise) based on the current game state
- Learn and improve its strategy over time through self-play and competition with other bot types

## Project Structure

```
RLPoker/
├── README.md
├── requirements.txt
└── src/
    ├── game.py                 # Game evaluation script
    ├── bots/                   # Bot implementations
    │   ├── __init__.py
    │   ├── allInBot.py
    │   ├── consolePlayer.py
    │   ├── cowardBot.py
    │   └── randomBot.py
    └── model/                  # RL implementation
        ├── __init__.py
        ├── config.py           # Configuration parameters
        ├── memory.py           # Experience replay buffer
        ├── model.py            # PPO algorithm implementation
        ├── networks.py         # Neural network architectures
        ├── parallel_train.py   # Multi-process training
        ├── plotting.py         # Visualization utilities
        ├── train.py            # Single-process training
        ├── utils.py            # Feature extraction utilities
        └── wrapper.py          # PyPokerEngine integration
```

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- PyPokerEngine
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RLPoker.git
   cd RLPoker
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Using venv
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a New Agent

To start training a new poker agent from scratch:

```bash
cd src
python -m model.train
```

For multi-process training (faster but more resource-intensive):

```bash
cd src
python -m model.parallel_train
```

### Evaluating Agent Performance

To evaluate your trained agent against other bots:

```bash
cd src
python game.py
```

This will run many simulated games between your trained agent and various opponent bots, then display the win statistics.

### Configuration

You can modify training parameters by editing the `src/model/config.py` file:

- Adjust hyperparameters like learning rates and PPO settings
- Change the neural network architecture
- Modify opponent types and game settings
- Set the number of training games

## How It Works

### State Representation

The agent represents the poker game state as a vector with 30 features, including:
- Encoded hole cards
- Encoded community cards
- Estimated win rate
- Stack sizes
- Pot sizes and pot odds
- Position information
- Board texture analysis

### Decision Making

The agent makes decisions using two neural networks:
1. **Actor (Policy) Network**: Determines action probabilities (fold, call, raise)
2. **Critic (Value) Network**: Estimates expected rewards for a given state

### Training Process

1. The agent plays games against opponents, collecting experiences
2. Experiences are stored in a memory buffer as (state, action, reward) tuples
3. When enough experiences are collected, the PPO algorithm updates both networks
4. The agent uses new policy to collect more experiences, repeating the cycle

## License

This repository is licensed under the MIT License.

## Acknowledgments

- PyPokerEngine framework
- PPO algorithm paper by Schulman et al.
- PyTorch framework
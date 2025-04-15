# ppo_core.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

try:
    from . import config
    from .networks import Actor, Critic
    from .memory import MemoryBuffer
except ImportError:
    # Fallback for running script directly
    import config
    from networks import Actor, Critic
    from memory import MemoryBuffer

class PPO:
    """Implements the Proximal Policy Optimization algorithm logic."""
    def __init__(self, device=config.DEVICE):
        self.device = device

        # Initialize Actor and Critic networks
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        # Initialize Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE_CRITIC)

        # Initialize Memory Buffer
        self.memory = MemoryBuffer(device=self.device)

        self.learn_step_counter = 0


    def select_action(self, state_np):
        """Selects action, gets log prob and value estimate for a single state."""
        if state_np is None or not isinstance(state_np, np.ndarray):
            # Handle cases where state might be invalid
            print("Warning: Received invalid state in select_action. Returning default action.")
            # Return dummy values (e.g., fold action, zero log_prob/value)
            # The caller should ideally prevent invalid states.
            action_idx = 0 # Fold
            log_prob = torch.tensor(0.0, device=self.device)
            value = torch.tensor(0.0, device=self.device)
            return action_idx, log_prob, value # Ensure tensors are returned

        state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device) # Add batch dim
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state)
            value = self.critic(state)

        # Store step data (will be moved to CPU in memory.store_step)
        self.memory.store_step(state_np, action, log_prob, value)

        return action.item(), log_prob, value # Return CPU tensors/values for storage if needed

    def finish_hand(self, final_reward):
        """Finalizes the current trajectory in memory."""
        self.memory.finish_trajectory(final_reward)

    def update(self):
        """Performs the PPO update step using data from memory."""
        if not self.memory.ready():
            # print("Skipping update: Not enough data in memory.")
            return False # Indicate update did not happen

        print(f"Starting PPO update #{self.learn_step_counter + 1} with {len(self.memory)} samples...")
        actor_losses, critic_losses, entropy_bonuses = [], [], []

        # Set networks to training mode
        self.actor.train()
        self.critic.train()

        # Iterate over experience batches multiple times (epochs)
        for _ in range(config.PPO_EPOCHS):
            batch_generator = self.memory.generate_batches() # Regenerates GAE/returns each epoch start

            for batch in batch_generator:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']

                # Normalize advantages (per mini-batch) - crucial for stability
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- Calculate Actor Loss (Policy Loss) ---
                # Evaluate current policy
                action_probs = self.actor(states) # Get new probabilities
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Calculate ratio: pi(a|s) / pi_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped Surrogate Objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.PPO_EPSILON, 1.0 + config.PPO_EPSILON) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Calculate Critic Loss (Value Loss) ---
                # Get current value estimates
                values = self.critic(states).squeeze() # Remove extra dim
                # Mean Squared Error loss against calculated returns (targets)
                critic_loss = F.mse_loss(values, returns)

                # --- Calculate Total Loss ---
                total_loss = actor_loss + config.VALUE_LOSS_COEFF * critic_loss - config.ENTROPY_BETA * entropy

                # --- Backpropagation and Optimization ---
                # Actor
                self.actor_optimizer.zero_grad()
                # Retain graph if critic loss depends on actor output (not typical here, but good practice)
                actor_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5) # Optional gradient clipping
                self.actor_optimizer.step()

                # Critic
                self.critic_optimizer.zero_grad()
                # Calculate critic loss again if necessary or use the one computed above
                critic_loss_term = config.VALUE_LOSS_COEFF * critic_loss # Scale before backward pass
                critic_loss_term.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5) # Optional gradient clipping
                self.critic_optimizer.step()

                # --- Logging (optional per mini-batch) ---
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append(entropy.item())


        # Clear memory now handled within generate_batches loop start
        # self.memory.clear_memory() # Memory is cleared when generate_batches is first called

        # Log average losses for the update step
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        avg_entropy = np.mean(entropy_bonuses)
        print(f"Update complete. Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}, Avg Entropy: {avg_entropy:.4f}")

        self.learn_step_counter += 1

        # Set networks back to evaluation mode
        self.actor.eval()
        self.critic.eval()

        return True # Indicate update happened


    def save_model(self, path_prefix):
        """Saves actor and critic models."""
        try:
            torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
            torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
            print(f"Models saved successfully to {path_prefix}_*.pth")
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_model(self, path_prefix):
        """Loads actor and critic models."""
        try:
            self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth", map_location=self.device))
            self.actor.to(self.device)
            self.critic.to(self.device)
            # Set to eval mode after loading
            self.actor.eval()
            self.critic.eval()
            print(f"Models loaded successfully from {path_prefix}_*.pth")
        except FileNotFoundError:
            print(f"Warning: Model files not found at {path_prefix}_*. Starting from scratch.")
        except Exception as e:
            print(f"Error loading models: {e}. Starting from scratch.")
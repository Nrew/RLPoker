import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


try:
    from . import config
    from .networks import Actor, Critic
    from .memory import MemoryBuffer
except ImportError:
    # Fallback import
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

        self.actions_file = open("logs/actions.txt", "a")


    def select_action(self, state_np):
        """Selects action, gets log prob and value estimate for a single state."""
        if state_np is None or not isinstance(state_np, np.ndarray):
            # Handle cases where state might be invalid
            print("Warning: Received invalid state in select_action. Returning default action.")
            
            # Return dummy values (e.g., fold action, zero log_prob/value)
            action_idx = 0 # Fold
            log_prob = torch.tensor(0.0, device=self.device)
            value = torch.tensor(0.0, device=self.device).unsqueeze(0)
            self.actions_file.write(f"{action_idx}\n")
            self.actions_file.flush()
            return action_idx, log_prob, value.squeeze()

        state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device) # Add batch dim
        
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state)
            value = self.critic(state)

        # Store step data (will be moved to CPU in memory.store_step)
        self.memory.store_step(state_np, action, log_prob, value)
        self.actions_file.write(f"{action}\n")
        self.actions_file.flush()

        return action.item(), log_prob, value # Return CPU tensors/values for storage if needed

    def finish_hand(self, final_reward):
        """Finalizes the current trajectory in memory."""
        self.memory.finish_trajectory(final_reward)

    def update(self):
        """Performs the PPO update step using data from memory."""
        if not self.memory.ready():
            return False # Indicate update did not happen

        print(f"Starting PPO update #{self.learn_step_counter + 1} with {len(self.memory)} samples...")
        actor_losses, critic_losses, entropy_bonuses = [], [], []

        try:
            batch_data = self.memory.prepare_update_data()
            n_samples = batch_data['n_samples']
        except RuntimeError as e:
            print(f"Error preparing update data: {e}")
            return False # Update impossible
    
        # --- Update loop ---
    
        # Set networks to training mode
        self.actor.train()
        self.critic.train()

        for epoch in range(config.PPO_EPOCHS):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, config.BATCH_SIZE):
                end_idx = min(start_idx + config.BATCH_SIZE, n_samples)
                if start_idx >= end_idx: continue

                minibatch_indices = indices[start_idx:end_idx]
                
                states = batch_data['states'][minibatch_indices]
                actions = batch_data['actions'][minibatch_indices]
                old_log_probs = batch_data['old_log_probs'][minibatch_indices]
                advantages = batch_data['advantages'][minibatch_indices]
                returns = batch_data['returns'][minibatch_indices] # V_target
                

                # Normalize advantages (per mini-batch) - crucial for stability
                advantages = (advantages - advantages.mean()) / (advantages.std() + config.EPSILON)

                # --- Calculate Actor Loss (Policy Loss) ---
                action_probs = self.actor(states)
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
                values = self.critic(states).squeeze(-1)
                critic_loss = F.mse_loss(values, returns)

                # --- Calculate Total Loss ---
                total_loss = actor_loss + config.VALUE_LOSS_COEFF * critic_loss - config.ENTROPY_BETA * entropy
                
                # --- Backpropagation and Optimization ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                

                # --- Logging (optional per mini-batch) ---
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append(entropy.item())

        # --- Update step complete ---
        self.learn_step_counter += 1

        avg_actor_loss = np.mean(actor_losses)   if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_entropy = np.mean(entropy_bonuses)   if entropy_bonuses else 0
        print(f"Update #{self.learn_step_counter} complete. Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}, Avg Entropy: {avg_entropy:.4f}")

        self.actor.eval()
        self.critic.eval()

        self.log_metrics(avg_entropy, avg_critic_loss, avg_actor_loss)

        return True

    @staticmethod
    def log_metrics(entropy, critic_loss, actor_loss):
        os.makedirs("logs", exist_ok=True)
        with open("logs/entropy.txt", "a") as entropy_file:
            entropy_file.write(f"{entropy}\n")

        with open("logs/critic_loss.txt", "a") as critic_loss_file:
            critic_loss_file.write(f"{critic_loss}\n")

        with open("logs/actor_loss.txt", "a") as actor_loss_file:
            actor_loss_file.write(f"{actor_loss}\n")

    def save_model(self, path_prefix, game_num):
        """Saves actor and critic models."""
        try:
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            print(f"Attempting to save to {path_prefix}")
            torch.save(self.actor.state_dict(), f"{path_prefix}_actor_{game_num}.pth")
            torch.save(self.critic.state_dict(), f"{path_prefix}_critic_{game_num}.pth")
            torch.save(self.actor.state_dict(), f"{path_prefix}_actor_latest.pth")
            print(f" Saved To {path_prefix}_actor_latest.pth")
            torch.save(self.critic.state_dict(), f"{path_prefix}_critic_latest.pth")
            print(f"Saved to {path_prefix}_critic_latest.pth")
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
            print(f"Models loaded successfully from {path_prefix}_actor_latest.pth")
        except FileNotFoundError:
            print(f"Warning: Model files not found at {path_prefix}_*. Starting from scratch.")
        except Exception as e:
            print(f"Error loading models: {e}. Starting from scratch.")

    def load_latest(self, path_prefix):
        """Loads the latest actor and critic models."""
        try:
            self.actor.load_state_dict(torch.load(f"{path_prefix}_actor_latest.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{path_prefix}_critic_latest.pth", map_location=self.device))
            self.actor.to(self.device)
            self.critic.to(self.device)
            # Set to eval mode after loading
            self.actor.eval()
            self.critic.eval()
            print(f"Models loaded successfully from {path_prefix}_latest.pth")
        except FileNotFoundError:
            print(f"Warning: Model files not found at {path_prefix}_*. Starting from scratch. (Load Latest)")
        except Exception as e:
            print(f"Error loading models: {e}. Starting from scratch. (Load Latest)")


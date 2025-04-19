import numpy as np
import torch

try:
    from . import config
except ImportError:
    import config

class MemoryBuffer:
    """Stores transitions for PPO update."""
    def __init__(self,
        batch_size=config.BATCH_SIZE,
        buffer_size=config.BUFFER_SIZE,
        device=config.DEVICE):
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        self.batch_size = batch_size
        self.buffer_size = buffer_size # Max transitions before forcing update
        self.device = device
        self._current_size = 0

        # Temp storage for current trajectory
        self._traj_states = []
        self._traj_actions = []
        self._traj_log_probs = []
        self._traj_values = []

    def store_step(self, state, action, log_prob, value):
        """Stores data for a single step within the current trajectory."""
        self._traj_states.append(state)  # Keep as numpy for now
        self._traj_actions.append(action.item() if isinstance(action, torch.Tensor) else int(action))
        self._traj_log_probs.append(log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob))
        value_item = value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value.cpu().detach().numpy().flatten()[0] if isinstance(value, torch.Tensor) else float(value)
        self._traj_values.append(value_item)
        # Ensure data is on CPU for storage in Python lists
        # self._traj_states.append(state) # Assuming state is already numpy array
        # self._traj_actions.append(action.cpu().item() if isinstance(action, torch.Tensor) else int(action))
        # self._traj_log_probs.append(log_prob.cpu().detach().numpy() if isinstance(log_prob, torch.Tensor) else float(log_prob))
        # value_np = value.cpu().detach().numpy().flatten()
        # self._traj_values.append(value_np[0] if value_np.size == 1 else 0.0)
   
    def finish_trajectory(self, final_reward):
        """
        Marks the end of a trajectory (hand), calculates per-step rewards,
        and adds the full trajectory to the main buffer.
        """
        if not self._traj_states:
            self._clear_trajectory_buffer()
            return # No steps taken

        traj_len = len(self._traj_states)
        # Simple reward distribution: assign final reward to the last step, 0 elsewhere
        # TODO: More sophisticated distribution or shaping.
        step_rewards = np.zeros(traj_len, dtype=np.float32)
        step_dones = np.zeros(traj_len, dtype=np.float32)

        if traj_len > 0:
            step_rewards[-1] = final_reward
            step_dones[-1] = 1.0 # Mark the last step as done

        # Append trajectory data to main buffer lists
        self.states.extend(self._traj_states)
        self.actions.extend(self._traj_actions)
        self.log_probs.extend(self._traj_log_probs)
        self.rewards.extend(step_rewards)
        self.values.extend(self._traj_values)
        self.dones.extend(step_dones)

        self._current_size += traj_len

        # Clear temporary trajectory storage
        self._clear_trajectory_buffer()
        
        # Buffer overflow check
        if self._current_size > self.buffer_size:
            num_to_remove = self._current_size - self.buffer_size
            
            # Remove at least one batch worth
            num_to_remove = max(self.batch_size, num_to_remove)
            
            # Don't remove more than exist
            num_to_remove = min(num_to_remove, self._current_size) 
            print(f"Memory buffer overflow. Removing {num_to_remove} oldest samples.")
            
            # Remove oldest samples from each list
            del self.states[:num_to_remove]
            del self.actions[:num_to_remove]
            del self.log_probs[:num_to_remove]
            del self.rewards[:num_to_remove]
            del self.values[:num_to_remove]
            del self.dones[:num_to_remove]
            self._current_size -= num_to_remove

    def _clear_trajectory_buffer(self):
        self._traj_states = []
        self._traj_actions = []
        self._traj_log_probs = []
        self._traj_values = []
        
    def prepare_update_data(self):
        """
        Extracts all data, calculates GAE/Returns, converts to tensors,
        clears memory, and returns the tensors.
        """
        if not self.ready():
            raise RuntimeError("prepare_update_data called with insufficient samples.")

        n_samples = self._current_size
        states_arr = np.array(self.states[:n_samples], dtype=np.float32)
        actions_arr = np.array(self.actions[:n_samples], dtype=np.int64)
        log_probs_arr = np.array(self.log_probs[:n_samples], dtype=np.float32).flatten()
        rewards_arr = np.array(self.rewards[:n_samples], dtype=np.float32).flatten()
        values_arr = np.array(self.values[:n_samples], dtype=np.float32).flatten()
        dones_arr = np.array(self.dones[:n_samples], dtype=np.float32).flatten()

        # Calculate GAE and Returns
        advantages, returns = self._calculate_gae(rewards_arr, values_arr, dones_arr)

        # Clear memory AFTER extracting data
        self.clear_memory()

        # Convert data to tensors
        batch_tensors = {
            'states': torch.tensor(states_arr, dtype=torch.float32).to(self.device),
            'actions': torch.tensor(actions_arr, dtype=torch.long).to(self.device),
            'old_log_probs': torch.tensor(log_probs_arr, dtype=torch.float32).to(self.device),
            'advantages': torch.tensor(advantages, dtype=torch.float32).to(self.device),
            'returns': torch.tensor(returns, dtype=torch.float32).to(self.device), # Value targets
            'n_samples': n_samples # Include sample count
        }
        return batch_tensors

    @DeprecationWarning
    def generate_batches(self):
        """
        Generates batches of experience data as PyTorch tensors.
        Clears the memory after generating.
        """
        if not self.ready():
            print("Warning: generate_batches called with insufficient data.")
            return iter([])
        
        # Convert lists to numpy arrays first for efficiency
        n_samples = self._current_size
        states_arr = np.array(self.states[:n_samples], dtype=np.float32)
        actions_arr = np.array(self.actions[:n_samples], dtype=np.int64)
        log_probs_arr = np.array(self.log_probs[:n_samples], dtype=np.float32).flatten() # Ensure 1D
        rewards_arr = np.array(self.rewards[:n_samples], dtype=np.float32).flatten()
        values_arr = np.array(self.values[:n_samples], dtype=np.float32).flatten()
        dones_arr = np.array(self.dones[:n_samples], dtype=np.float32).flatten()

        # Calculate advantages and returns using GAE
        advantages, returns = self._calculate_gae(rewards_arr, values_arr, dones_arr)

        # Create tensors and move to device
        batch = {
            'states': torch.tensor(states_arr, dtype=torch.float32).to(self.device),
            'actions': torch.tensor(actions_arr, dtype=torch.long).to(self.device),
            'old_log_probs': torch.tensor(log_probs_arr, dtype=torch.float32).to(self.device),
            'advantages': torch.tensor(advantages, dtype=torch.float32).to(self.device),
            'returns': torch.tensor(returns, dtype=torch.float32).to(self.device)
        }

        # Clear memory after generating batches
        self.clear_memory()

        # Shuffle indices for batch iteration during PPO epochs
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # Yield mini-batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            if start_idx >= end_idx: continue 
            minibatch_indices = indices[start_idx:end_idx]
            yield {key: tensor[minibatch_indices] for key, tensor in batch.items()}

    def _calculate_gae(self, rewards, values, dones):
        """Calculates Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0.0
        n_steps = len(rewards)

        # NOTE: Need next state value estimate for delta calculation
        # Append a 0 value for the terminal state if the last state wasn't terminal,
        # or use the value estimate of the last state if it *was* terminal (should be 0).
        # TODO: (FIX) TEMP Approach: Use V(s_t+1) where available, 0 if done[t] == 1.
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t] # 1 if not done, 0 if done
                next_value = 0                     # No V(s_{t+1}) available
            else:
                next_non_terminal = 1.0 - dones[t] # Is the *current* step terminal?
                next_value = values[t + 1]         # V(s_{t+1})

            delta = rewards[t] + config.GAMMA * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + config.GAMMA * config.GAE_LAMBDA * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values # TD(lambda) returns = GAE + V(s)
        return advantages, returns

    def ready(self):
        """Check if enough data is collected for a batch update."""
        return self._current_size >= self.batch_size

    def is_full(self):
        """Check if the buffer size limit is reached."""
        return self._current_size >= self.buffer_size

    def clear_memory(self):
        """Resets the main storage buffers."""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]
        self._current_size = 0
        self._clear_trajectory_buffer() # Clear any incomplete trajectory

    def __len__(self):
        return self._current_size
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# --- Hyperparameters ---

# The MuJoCo environment name
# The PPO paper used: "HalfCheetah-v5", "Hopper-v5", "InvertedDoublePendulum-v5", "InvertedPendulum-v5", "Reacher-v5", "Swimmer-v5", "Walker2d-v5"
# The mb-mf paper used: "Ant-v5", "HalfCheetah-v5", "Hopper-v5" and "Swimmer-v5"
# For the comparison I will use only "HalfCheetah-v5", "Hopper-v5" and "Swimmer-v5"
env_id: str = "HalfCheetah-v5" # S ∈ R23, A ∈ R6
env_id: str = "Hopper-v5" # S ∈ R17, A ∈ R3
env_id: str = "Swimmer-v5" # S ∈ R16, A ∈ R2
seed: int = 1
cuda: bool = True

# Phase 1: Model-Based args
mb_random_timesteps: int = 5000  # Initial random rollout steps to seed dynamics model
mb_mpc_timesteps: int = 15000    # Timesteps to run MPC and gather good data
dyn_epochs: int = 10             # Dynamics model training epochs per iteration
dyn_batch_size: int = 512
mpc_horizon: int = 10            # H: Horizon for random shooting
mpc_num_sequences: int = 1000    # K: Number of random action sequences to sample

# Phase 2: Cloning args
cloning_epochs: int = 10         # number of loops through the entire dataset (state MPC-action pairs)
cloning_batch_size: int = 256    # number of (state, action) pairs that are grouped together to calculate the error before the NN updates its weights.

# Phase 3: Model-Free (PPO) Fine-Tuning args
mf_total_timesteps: int = 1000000 # total timesteps of the experiments
learning_rate: float = 3e-4       # learning rate of the optimizer
num_steps: int = 2048             # horizon: the number of steps the agent takes in each environment before it stops to learn
gamma: float = 0.99               # discount factor: determines how much the agent values future rewards compared to immediate rewards
gae_lambda: float = 0.95          # 0.0  (Low Variance, High Bias): immediate 1-step reward plus the Value Network's guess for the next state
                                  # 1.0  (High Variance, No Bias): sums up all the actual rewards until the end of the episode
update_epochs: int = 10           # the number (K) of epochs to update the policy
num_minibatches: int = 32         # the number of mini-batches
clip_coef: float = 0.2            # the surrogate clipping coefficient. 0.2 should be the best regarding the PPO paper.
entropy_coef: float = 0.0         # coefficient of the entropy controls the weight of the entropy bonus added to the training loss function
# the ppo paper used 0.1 for discrete action spaces, to support exploration and prevent the agent's policy from collapsing into a suboptimal strategy.
# the ppo paper used 0.0 for continuous action spaces, because the continuous Normal distribution's variance already regulates exploration
vf_coef: float = 0.5              # coefficient of the value function controls the relative importance of the value function loss 
max_grad_norm: float = 0.5        # the maximum norm for the gradient clipping controls


# --- Dynamics Model ---
class DynamicsModel(nn.Module):
    """
    Predicts the state differences: s_{t+1} - s_t = f(s_t, a_t)
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# --- MPC Controller ---
class MPCController:
    """
    Random shooting Model Predictive Control (MPC) using the learned dynamics model.
    """
    def __init__(self, env, dynamics_model, horizon, num_sequences, device):
        self.env = env
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.num_sequences = num_sequences
        self.device = device
        self.action_dim = np.prod(env.unwrapped.action_space.shape)
        self.action_low = env.unwrapped.action_space.low
        self.action_high = env.unwrapped.action_space.high

    def get_action(self, state):
        self.dynamics.eval()
        
        # 1. Sample K random action sequences over horizon H
        # Shape: (K, H, A)
        action_seqs = np.random.uniform(
            low=self.action_low, 
            high=self.action_high, 
            size=(self.num_sequences, self.horizon, self.action_dim)
        )
        action_seqs_tensor = torch.FloatTensor(action_seqs).to(self.device)
        
        # 2. Simulate trajectories
        states = torch.FloatTensor(state).unsqueeze(0).repeat(self.num_sequences, 1).to(self.device)
        total_rewards = torch.zeros(self.num_sequences).to(self.device)
        
        with torch.no_grad():
            for t in range(self.horizon):
                actions = action_seqs_tensor[:, t, :]
                state_diffs = self.dynamics(states, actions)
                next_states = states + state_diffs
                rewards = self.heuristic_reward(states, actions, next_states)
                total_rewards += rewards
                
                states = next_states
                
        # 3. Pick the first action of the sequence with the highest total reward
        best_seq_idx = torch.argmax(total_rewards).item()
        best_action = action_seqs[best_seq_idx, 0, :]
        return best_action

    # got the heuristics from https://gymnasium.farama.org/environments/mujoco/
    def heuristic_reward(self, state, action, next_state):
        if env_id == "HalfCheetah-v5":
            # v_x is located at index 8 of the observation space
            forward_reward = next_state[:, 8]
            # Control cost weight for HalfCheetah is 0.1
            ctrl_cost = 0.1 * torch.sum(torch.square(action), dim=-1)
            return forward_reward - ctrl_cost # vₓ - (0.1 × ||action||²₂)

        elif env_id == "Hopper-v5":
            z_height = next_state[:, 0]
            angle = next_state[:, 1]
            forward_reward = next_state[:, 5]  # v_x is at index 5
            # Evaluate healthy condition (not fallen over, not tilted too far)
            is_healthy = (z_height > 0.7) & (torch.abs(angle) < 0.2)
            healthy_reward = 1.0 * is_healthy.float()
            # Control cost weight for Hopper is 1e-3
            ctrl_cost = 0.001 * torch.sum(torch.square(action), dim=-1)
            return forward_reward + healthy_reward - ctrl_cost # vₓ + 1.0(if healthy) - (0.001 × ||action||²₂)

        elif env_id == "Swimmer-v5":
            # v_x is located at index 3 of the observation space
            forward_reward = next_state[:, 3] 
            # Control cost weight for Swimmer is 1e-4
            ctrl_cost = 0.0001 * torch.sum(torch.square(action), dim=-1)
            return forward_reward - ctrl_cost # vₓ - (0.0001 × ||action||²₂)
        
        else:
            return torch.zeros(state.shape[0]).to(self.device)


# --- Model-Free Agent (PPO / TRPO style) ---
class Agent(nn.Module):
    """
    Standard Actor-Critic for Model-Free Fine-tuning, cloned from your PPO implementation.
    """
    def __init__(self, env):
        super().__init__()
        state_dim = np.array(env.observation_space.shape).prod()
        action_dim = np.prod(env.action_space.shape)
        
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -1.0))

    def get_action(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd)
        return Normal(mean, std)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # Same as in your ppo.py
        probs = self.get_action(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# --- Training Loops ---
def train_dynamics(dynamics, optimizer, replay_buffer, epochs, batch_size, device):
    dynamics.train()
    states, actions, next_states = replay_buffer
    
    # The target is the residual (state difference)
    targets = next_states - states
    
    # Ideally, you should normalize states, actions, and targets for stable training.
    dataset = torch.utils.data.TensorDataset(states, actions, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for b_states, b_actions, b_targets in dataloader:
            b_states, b_actions, b_targets = b_states.to(device), b_actions.to(device), b_targets.to(device)
            
            optimizer.zero_grad()
            preds = dynamics(b_states, b_actions)
            loss = F.mse_loss(preds, b_targets)
            loss.backward()
            optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ProximalPolicyOptimization.ppo import init_gymnasium
    
    run_name = f"MBMF_{env_id}_{seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    env = init_gymnasium(env_id, capture_video=False, run_name=run_name, gamma=gamma)
    
    writer = SummaryWriter(f"runs/{run_name}")
    mb_writer = SummaryWriter(f"runs/{run_name}_Phase1_MB")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([
            f"|env_id|{env_id}|",
            f"|seed|{seed}|",
            f"|mb_random_timesteps|{mb_random_timesteps}|",
            f"|mb_mpc_timesteps|{mb_mpc_timesteps}|",
            f"|dyn_epochs|{dyn_epochs}|",
            f"|dyn_batch_size|{dyn_batch_size}|",
            f"|mpc_horizon|{mpc_horizon}|",
            f"|mpc_num_sequences|{mpc_num_sequences}|",
            f"|cloning_epochs|{cloning_epochs}|",
            f"|cloning_batch_size|{cloning_batch_size}|",
            f"|mf_total_timesteps|{mf_total_timesteps}|",
            f"|learning_rate|{learning_rate}|",
            f"|num_steps|{num_steps}|",
            f"|gamma|{gamma}|",
            f"|gae_lambda|{gae_lambda}|",
            f"|update_epochs|{update_epochs}|",
            f"|num_minibatches|{num_minibatches}|",
            f"|clip_coef|{clip_coef}|",
            f"|entropy_coef|{entropy_coef}|",
            f"|vf_coef|{vf_coef}|",
            f"|max_grad_norm|{max_grad_norm}|",
        ]),
    )
    
    state_dim = np.array(env.observation_space.shape).prod()
    action_dim = np.prod(env.action_space.shape)
    
    # Initialize Models
    dynamics_model = DynamicsModel(state_dim, action_dim).to(device)
    dyn_optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    
    agent = Agent(env).to(device)
    agent_optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    
    mpc = MPCController(env, dynamics_model, mpc_horizon, mpc_num_sequences, device)
    
    # Storage for Replay Buffer
    states_buf, actions_buf, next_states_buf = [], [], []

    # =================================================================================
    # Phase 1: Model-Based Pre-training (Train Dynamics & Gather MPC Data)
    # =================================================================================
    print("=== Phase 1: Model-Based Data Collection & Dynamics Training ===")
    obs, _ = env.reset(seed=seed)
    
    for t in range(mb_random_timesteps + mb_mpc_timesteps):
        if t < mb_random_timesteps:
            action = env.unwrapped.action_space.sample()  # Random actions for initial exploration
        else:
            action = mpc.get_action(obs)        # MPC actions
            
        next_obs, reward, terminated, truncated, infos = env.step(action)
        
        states_buf.append(obs)
        actions_buf.append(action)
        next_states_buf.append(next_obs)
        
        obs = next_obs
        
        if "episode" in infos:
            print(f"Phase 1 (Step {t}): episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], t)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], t)
            mb_writer.add_scalar("charts/episodic_return", infos["episode"]["r"], t)
            mb_writer.add_scalar("charts/episodic_length", infos["episode"]["l"], t)
            
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
            
        # Periodically train dynamics model
        if t >= mb_random_timesteps and t % 1000 == 0:
            print(f"Training Dynamics Model at step {t}")
            rb = (torch.FloatTensor(np.array(states_buf)), 
                  torch.FloatTensor(np.array(actions_buf)), 
                  torch.FloatTensor(np.array(next_states_buf)))
            train_dynamics(dynamics_model, dyn_optimizer, rb, dyn_epochs, dyn_batch_size, device)


    # =================================================================================
    # Phase 2: Behavioral Cloning (Initialize Policy from MPC)
    # =================================================================================
    print("\n=== Phase 2: Behavioral Cloning (Initialize Policy) ===")
    print("Cloning MPC actions to initialize the Model-Free policy...")
    
    # We use the states and actions gathered to supervise the actor network
    cloning_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(np.array(states_buf)), 
        torch.FloatTensor(np.array(actions_buf))
    )
    cloning_loader = torch.utils.data.DataLoader(cloning_dataset, batch_size=cloning_batch_size, shuffle=True)
    
    for epoch in range(cloning_epochs):
        epoch_loss = 0
        for b_states, b_actions in cloning_loader:
            b_states, b_actions = b_states.to(device), b_actions.to(device)
            
            # Actor Loss (MSE) - Leaves the standard deviation untouched for PPO exploration!
            predicted_actions = agent.actor_mean(b_states)
            loss = F.mse_loss(predicted_actions, b_actions)
            
            agent_optimizer.zero_grad()
            loss.backward()
            agent_optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Cloning Epoch {epoch+1}/{cloning_epochs} | Loss: {epoch_loss/len(cloning_loader):.4f}")


    # =================================================================================
    # Phase 3: Model-Free Fine-Tuning (PPO)
    # =================================================================================
    print("\n=== Phase 3: Model-Free Fine-Tuning (PPO) ===")
    print("Starting Model-Free Fine-Tuning...")
    
    # Import train_ppo from your PPO implementation
    from ProximalPolicyOptimization.ppo import train_ppo
    
    batch_size = int(num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = mf_total_timesteps // batch_size
    
    # We pass the same writer and run_name so PPO fine-tuning logs to the same Tensorboard run
    train_ppo(
        env=env, 
        agent=agent, 
        optimizer=agent_optimizer, 
        device=device, 
        writer=writer, 
        num_steps=num_steps, 
        num_iterations=num_iterations, 
        batch_size=batch_size, 
        minibatch_size=minibatch_size, 
        update_epochs=update_epochs, 
        learning_rate=learning_rate, 
        anneal_lr=True, 
        gamma=gamma, 
        gae_lambda=gae_lambda, 
        clip_coef=clip_coef, 
        normalize_advantage=True, 
        entropy_coef=entropy_coef, 
        vf_coef=vf_coef, 
        max_grad_norm=max_grad_norm, 
        target_kl=None, 
        save_model=True, 
        run_name=run_name, 
        seed=seed,
        global_step=mb_random_timesteps + mb_mpc_timesteps
    )
    
    print("Draft complete!")

if __name__ == "__main__":
    main()

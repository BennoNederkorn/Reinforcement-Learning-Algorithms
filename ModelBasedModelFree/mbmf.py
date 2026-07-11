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
if env_id == "Swimmer-v5":
    mb_random_timesteps: int = 200 * 500  # TF
    tf_dyn_epochs: int = 70
    aggregation_iters: int = 6
    rollouts_per_iter: int = 9
    rollout_length: int = 333
    f_dyn_epochs: int = 30
    mpc_horizon: int = 20            # H
    mpc_num_sequences: int = 5000    # K
    # Phase 2: BC args
    bc_initial_rollouts: int = 30
    bc_dagger_iters: int = 3
    bc_dagger_epochs: int = 70
    bc_rollouts_per_iter: int = 5
elif env_id == "HalfCheetah-v5":
    mb_random_timesteps: int = 200 * 1000 # TF
    tf_dyn_epochs: int = 40
    aggregation_iters: int = 7
    rollouts_per_iter: int = 9
    rollout_length: int = 1000
    f_dyn_epochs: int = 60
    mpc_horizon: int = 20            # H
    mpc_num_sequences: int = 1000    # K
    # Phase 2: BC args
    bc_initial_rollouts: int = 30
    bc_dagger_iters: int = 3
    bc_dagger_epochs: int = 300
    bc_rollouts_per_iter: int = 2
elif env_id == "Hopper-v5":
    mb_random_timesteps: int = 20 * 200   # TF
    tf_dyn_epochs: int = 40
    aggregation_iters: int = 5
    rollouts_per_iter: int = 10
    rollout_length: int = 200
    f_dyn_epochs: int = 40
    mpc_horizon: int = 40            # H
    mpc_num_sequences: int = 1000    # K
    # Phase 2: BC args
    bc_initial_rollouts: int = 60
    bc_dagger_iters: int = 5
    bc_dagger_epochs: int = 200
    bc_rollouts_per_iter: int = 5
else:
    mb_random_timesteps: int = 5000       # Default
    tf_dyn_epochs: int = 40
    aggregation_iters: int = 5
    rollouts_per_iter: int = 10
    rollout_length: int = 300
    f_dyn_epochs: int = 40
    mpc_horizon: int = 10
    mpc_num_sequences: int = 1000
    bc_initial_rollouts: int = 10
    bc_dagger_iters: int = 2
    bc_dagger_epochs: int = 50
    bc_rollouts_per_iter: int = 2

mb_mpc_timesteps: int = aggregation_iters * rollouts_per_iter * rollout_length
dyn_epochs: int = f_dyn_epochs  # Fallback for old variables
dyn_batch_size: int = 512

# Phase 2: Cloning args
cloning_lr: float = 1e-4
cloning_batch_size: int = 500

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
target_kl: float = None           # Emulate TRPO's strict KL divergence constraint


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
        
        # Normalization parameters
        self.register_buffer('state_mean', torch.zeros(1, state_dim))
        self.register_buffer('state_std', torch.ones(1, state_dim))
        self.register_buffer('action_mean', torch.zeros(1, action_dim))
        self.register_buffer('action_std', torch.ones(1, action_dim))
        self.register_buffer('delta_mean', torch.zeros(1, state_dim))
        self.register_buffer('delta_std', torch.ones(1, state_dim))
        
    def forward(self, state, action, return_normalized=False):
        norm_state = (state - self.state_mean) / (self.state_std + 1e-8)
        norm_action = (action - self.action_mean) / (self.action_std + 1e-8)
        
        x = torch.cat([norm_state, norm_action], dim=-1)
        norm_delta = self.net(x)
        
        if return_normalized:
            return norm_delta
            
        delta = (norm_delta * self.delta_std) + self.delta_mean
        return delta


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
    
    # Update normalization statistics
    dynamics.state_mean.data = states.mean(dim=0, keepdim=True).to(device)
    dynamics.state_std.data = states.std(dim=0, keepdim=True).to(device)
    
    dynamics.action_mean.data = actions.mean(dim=0, keepdim=True).to(device)
    dynamics.action_std.data = actions.std(dim=0, keepdim=True).to(device)
    
    dynamics.delta_mean.data = targets.mean(dim=0, keepdim=True).to(device)
    dynamics.delta_std.data = targets.std(dim=0, keepdim=True).to(device)
    
    # Normalize targets for computing MSE loss properly across all dimensions
    norm_targets = (targets.to(device) - dynamics.delta_mean) / (dynamics.delta_std + 1e-8)
    
    dataset = torch.utils.data.TensorDataset(states.to(device), actions.to(device), norm_targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for b_states, b_actions, b_norm_targets in dataloader:
            optimizer.zero_grad()
            # Predict normalized delta
            norm_preds = dynamics(b_states, b_actions, return_normalized=True)
            loss = F.mse_loss(norm_preds, b_norm_targets)
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
            f"|tf_dyn_epochs|{tf_dyn_epochs}|",
            f"|aggregation_iters|{aggregation_iters}|",
            f"|rollouts_per_iter|{rollouts_per_iter}|",
            f"|rollout_length|{rollout_length}|",
            f"|f_dyn_epochs|{f_dyn_epochs}|",
            f"|mpc_horizon|{mpc_horizon}|",
            f"|mpc_num_sequences|{mpc_num_sequences}|",
            f"|bc_initial_rollouts|{bc_initial_rollouts}|",
            f"|bc_dagger_iters|{bc_dagger_iters}|",
            f"|bc_dagger_epochs|{bc_dagger_epochs}|",
            f"|bc_rollouts_per_iter|{bc_rollouts_per_iter}|",
            f"|mb_mpc_timesteps|{mb_mpc_timesteps}|",
            f"|dyn_epochs|{dyn_epochs}|",
            f"|dyn_batch_size|{dyn_batch_size}|",
            f"|cloning_lr|{cloning_lr}|",
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
    
    # 1. Collect Initial Random Dataset (TF)
    print("Collecting Initial Random Dataset (D_RAND)...")
    obs, _ = env.reset(seed=seed)
    for t in range(mb_random_timesteps):
        action = env.unwrapped.action_space.sample()  # Random actions for initial exploration
        next_obs, reward, terminated, truncated, infos = env.step(action)
        
        states_buf.append(obs)
        actions_buf.append(action)
        next_states_buf.append(next_obs)
        obs = next_obs
        
        if "episode" in infos:
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], t)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], t)
            mb_writer.add_scalar("charts/episodic_return", infos["episode"]["r"], t)
            mb_writer.add_scalar("charts/episodic_length", infos["episode"]["l"], t)
            
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
            
    # 2. Train Dynamics on D_RAND
    print(f"Training Dynamics Model on D_RAND (TF) for {tf_dyn_epochs} epochs...")
    rb_rand = (torch.FloatTensor(np.array(states_buf)), 
               torch.FloatTensor(np.array(actions_buf)), 
               torch.FloatTensor(np.array(next_states_buf)))
    train_dynamics(dynamics_model, dyn_optimizer, rb_rand, tf_dyn_epochs, dyn_batch_size, device)
    
    # 3. DAgger MPC Aggregation Iterations (F)
    print("Starting MPC Aggregation Iterations (F)...")
    global_t = mb_random_timesteps
    for iter in range(aggregation_iters):
        print(f"--- Aggregation Iteration {iter+1}/{aggregation_iters} ---")
        for r in range(rollouts_per_iter):
            obs, _ = env.reset(seed=seed+iter+r)
            for step in range(rollout_length):
                action = mpc.get_action(obs)
                # Add exploration noise N(0, 0.005)
                action += np.random.normal(0, 0.005, size=action.shape)
                action = np.clip(action, env.unwrapped.action_space.low, env.unwrapped.action_space.high)
                
                next_obs, reward, terminated, truncated, infos = env.step(action)
                
                states_buf.append(obs)
                actions_buf.append(action)
                next_states_buf.append(next_obs)
                obs = next_obs
                global_t += 1
                
                if "episode" in infos:
                    print(f"Phase 1 (Step {global_t}): episodic_return={infos['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_t)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_t)
                    mb_writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_t)
                    mb_writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_t)
                if terminated or truncated:
                    break
        
        # 4. Train Dynamics on 10-90 Split
        d_rl_size = len(states_buf) - mb_random_timesteps
        num_d_rand_samples = max(1, int(d_rl_size / 9))
        d_rand_indices = np.random.choice(mb_random_timesteps, size=num_d_rand_samples, replace=True)
        d_rl_indices = np.arange(mb_random_timesteps, len(states_buf))
        train_indices = np.concatenate([d_rand_indices, d_rl_indices])
        
        print(f"Training Dynamics Model on Aggregated Dataset (F) for {f_dyn_epochs} epochs...")
        rb_agg = (torch.FloatTensor(np.array(states_buf)[train_indices]), 
                  torch.FloatTensor(np.array(actions_buf)[train_indices]), 
                  torch.FloatTensor(np.array(next_states_buf)[train_indices]))
        train_dynamics(dynamics_model, dyn_optimizer, rb_agg, f_dyn_epochs, dyn_batch_size, device)


    # =================================================================================
    # Phase 2: Behavioral Cloning (Initialize Policy from MPC)
    # =================================================================================
    print("\n=== Phase 2: Behavioral Cloning (Initialize Policy) ===")
    print("Cloning MPC actions to initialize the Model-Free policy using DAgger...")
    
    # The paper used a specific number of initial rollouts. Since we already collected D_RL in Phase 1,
    # we'll slice the last N rollouts to form the initial BC dataset.
    init_steps = bc_initial_rollouts * rollout_length
    bc_states = list(np.array(states_buf)[-init_steps:])
    bc_actions = list(np.array(actions_buf)[-init_steps:])
    
    cloning_optimizer = optim.Adam(agent.parameters(), lr=cloning_lr)
    
    for dagger_iter in range(bc_dagger_iters):
        print(f"--- Phase 2 DAgger Iteration {dagger_iter+1}/{bc_dagger_iters} ---")
        
        # Train Actor on current D_BC
        cloning_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.array(bc_states)), 
            torch.FloatTensor(np.array(bc_actions))
        )
        cloning_loader = torch.utils.data.DataLoader(cloning_dataset, batch_size=cloning_batch_size, shuffle=True)
        
        print(f"Training Behavioral Cloning for {bc_dagger_epochs} epochs...")
        for epoch in range(bc_dagger_epochs):
            epoch_loss = 0
            for b_states, b_actions in cloning_loader:
                b_states, b_actions = b_states.to(device), b_actions.to(device)
                
                cloning_optimizer.zero_grad()
                pred_actions = agent.actor_mean(b_states)
                loss = F.mse_loss(pred_actions, b_actions)
                loss.backward()
                cloning_optimizer.step()
                epoch_loss += loss.item()
                
            if (epoch + 1) % 10 == 0 or epoch == bc_dagger_epochs - 1:
                print(f"Epoch {epoch+1}/{bc_dagger_epochs}, Loss: {epoch_loss/len(cloning_loader):.6f}")
        
        # Rollout Actor to collect new states, then ask MPC for expert actions
        if bc_rollouts_per_iter > 0 and dagger_iter < bc_dagger_iters - 1:
            print(f"Running Actor for {bc_rollouts_per_iter} rollouts to collect DAgger queries...")
            agent.eval()
            for r in range(bc_rollouts_per_iter):
                obs, _ = env.reset(seed=seed+100+dagger_iter+r)
                for step in range(rollout_length):
                    with torch.no_grad():
                        actor_action = agent.actor_mean(torch.FloatTensor(obs).to(device)).cpu().numpy()
                    
                    next_obs, reward, terminated, truncated, infos = env.step(actor_action)
                    global_t += 1
                    
                    if "episode" in infos:
                        print(f"Phase 2 (Step {global_t}): episodic_return={infos['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_t)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_t)
                        mb_writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_t)
                        mb_writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_t)
                    
                    # Ask EXPERT (MPC) what it would have done
                    expert_action = mpc.get_action(obs)
                    expert_action += np.random.normal(0, 0.005, size=expert_action.shape)
                    expert_action = np.clip(expert_action, env.unwrapped.action_space.low, env.unwrapped.action_space.high)
                    
                    bc_states.append(obs)
                    bc_actions.append(expert_action)
                    
                    obs = next_obs
                    if terminated or truncated:
                        break
            agent.train()


    # =================================================================================
    # Phase 3: Model-Free Fine-Tuning (PPO)
    # =================================================================================
    print("\n=== Phase 3: Model-Free Fine-Tuning (PPO) ===")
    print("Starting Model-Free Fine-Tuning...")
    
    # Import train_ppo from your PPO implementation
    from ProximalPolicyOptimization.ppo import train_ppo
    
    batch_size = int(num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    
    # Calculate how many timesteps we have left to reach EXACTLY 1,000,000 total steps
    remaining_timesteps = mf_total_timesteps - global_t
    num_iterations = max(1, remaining_timesteps // batch_size)
    
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
        target_kl=target_kl, 
        save_model=True, 
        run_name=run_name, 
        seed=seed,
        global_step=global_t
    )
    
    print("Draft complete!")

if __name__ == "__main__":
    main()



# Conclusion: Initializing a model-free algorithm using a short-sighted Model-Based expert can actually permanently
# handicap the agent by throwing it into a deep local optimum that is too hard to unlearn. This is a very common finding
# when comparing Imitation Learning against from-scratch Reinforcement Learning!

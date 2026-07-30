import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import the PPO building blocks so both arms of the comparison (pure PPO vs. MB-MF)
# use literally the same network definition, initialization and training loop. Any
# difference between the two arms must come from the warm start, not from the code.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ProximalPolicyOptimization.ppo import Agent, init_gymnasium, train_ppo

# --- Hyperparameters ---

# The MuJoCo environment name
# The PPO paper used: "HalfCheetah-v5", "Hopper-v5", "InvertedDoublePendulum-v5", "InvertedPendulum-v5", "Reacher-v5", "Swimmer-v5", "Walker2d-v5"
# The mb-mf paper used: "Ant-v5", "HalfCheetah-v5", "Hopper-v5" and "Swimmer-v5"
# For the comparison I will use only "HalfCheetah-v5", "Hopper-v5" and "Swimmer-v5"
env_id: str = "HalfCheetah-v5" # S ∈ R23, A ∈ R6
# env_id: str = "Hopper-v5" # S ∈ R17, A ∈ R3
# env_id: str = "Swimmer-v5" # S ∈ R16, A ∈ R2
seed: int = 1
cuda: bool = True
num_envs: int = 8

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

# `rollout_length` is the DEPTH of one rollout (as in the MB-MF paper), not a sample
# budget to be split across parallel environments. With num_envs running in parallel,
# each pass of the rollout loop already produces num_envs complete rollouts, so the
# number of sequential passes is scaled down instead of the depth.
rollouts_per_pass: int = num_envs
# Rounded to the NEAREST number of passes so the realized rollout count stays as close as
# possible to the paper's `rollouts_per_iter` (with 8 envs, 9 requested rollouts -> 1 pass
# = 8 rollouts; rounding up instead would collect 16 and nearly double the sample budget).
sequential_rollout_passes: int = max(1, round(rollouts_per_iter / rollouts_per_pass))
mb_mpc_timesteps: int = aggregation_iters * sequential_rollout_passes * rollout_length * num_envs
dyn_batch_size: int = 512

# Phase 2: Cloning args
cloning_lr: float = 1e-4
cloning_batch_size: int = 500

# Phase 2.5: Critic warm-up args
# The warm-up must cover at least one full episode per environment, otherwise no episode
# ever completes and every Monte-Carlo return is truncated at an arbitrary cutoff.
warmup_steps_per_env: int = 1000
# Standard deviation the actor is given when PPO takes over. exp(-0.5) = 0.61, NARROWER
# than PPO's own from-scratch default of exp(0) = 1.0. A warm-started policy should
# exploit what behavioural cloning taught it; with std >= 1.0 on a [-1,1] action range
# more than half of all sampled actions saturate at the clip bound and the cloned mean
# is washed out entirely.
warmup_logstd: float = -0.5
# Rounds of fitted value iteration for the critic (targets are recomputed with the
# improved critic each round, so the bootstrap at truncation gets progressively better).
warmup_fit_rounds: int = 3
warmup_epochs: int = 20

# Phase 3: Model-Free (PPO) Fine-Tuning args
mf_total_timesteps: int = 1000000 # total timesteps of the experiments
learning_rate: float = 3e-4       # learning rate of the optimizer
num_steps: int = 2048 // num_envs # horizon: the number of steps the agent takes in each environment before it stops to learn
# MUST match ppo.py's per-environment choice, otherwise the MB-MF arm and the PPO
# baseline run different algorithms and the comparison measures the discount factor
# rather than the warm start. Swimmer needs the long horizon (see ppo.py).
gamma: float = 0.9999 if env_id.startswith("Swimmer") else 0.99
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
target_kl: float = 0.015           # Emulate TRPO's strict KL divergence constraint


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
        if hasattr(env, "envs"):
            act_space = env.envs[0].action_space
        elif hasattr(env, "single_action_space"):
            act_space = env.single_action_space
        else:
            act_space = env.unwrapped.action_space
        self.action_dim = np.prod(act_space.shape)
        self.action_low = np.nan_to_num(act_space.low, nan=-1.0, posinf=1.0, neginf=-1.0)
        self.action_high = np.nan_to_num(act_space.high, nan=1.0, posinf=1.0, neginf=-1.0)

    def get_actions(self, states):
        """Plan one action for EACH of the N states in a single batched pass.

        Planning for all parallel environments together turns N*H separate GPU calls
        into H calls on an N*K batch. The rollouts are full-depth (see the paper's
        rollout_length), so this batching is what keeps Phase 1 affordable.

        states: (N, state_dim) raw physical states.  Returns: (N, action_dim).
        """
        self.dynamics.eval()
        states = np.asarray(states)
        N, K, H, A = states.shape[0], self.num_sequences, self.horizon, self.action_dim

        # 1. Sample K random action sequences of horizon H, independently per state.
        action_seqs = np.random.uniform(
            low=self.action_low, high=self.action_high, size=(N, K, H, A)
        )
        action_seqs_tensor = torch.FloatTensor(action_seqs).to(self.device)

        # 2. Simulate all N*K candidate trajectories through the learned model.
        sim_states = (
            torch.FloatTensor(states).to(self.device)
            .unsqueeze(1).expand(N, K, states.shape[1]).reshape(N * K, states.shape[1])
        )
        total_rewards = torch.zeros(N * K, device=self.device)

        with torch.no_grad():
            for t in range(H):
                actions = action_seqs_tensor[:, :, t, :].reshape(N * K, A)
                next_states = sim_states + self.dynamics(sim_states, actions)
                total_rewards += self.heuristic_reward(sim_states, actions, next_states)
                sim_states = next_states

        # 3. For each state, take the first action of its best-scoring sequence.
        best_idx = torch.argmax(total_rewards.reshape(N, K), dim=1).cpu().numpy()
        return action_seqs[np.arange(N), best_idx, 0, :]

    def get_action(self, state):
        """Plan for a single state. Thin wrapper around the batched path."""
        return self.get_actions(np.asarray(state)[None, :])[0]

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


# The model-free agent is `Agent` imported from ppo.py -- deliberately NOT redefined here.
# A local copy previously used PyTorch's default initialization instead of PPO's
# orthogonal scheme, which made the critic start from a different distribution than the
# baseline and quietly turned the network initialization into a second uncontrolled
# variable in the comparison.


def get_raw_obs_vector(env):
    """Raw physical observations, bypassing the NormalizeObservation wrapper.

    The dynamics model and MPC operate in the true state space (that is where the
    heuristic reward's hand-indexed velocities live), while the policy sees the
    normalized observations the wrappers produce. `init_gymnasium` always returns a
    vector env, so this always returns shape (num_envs, state_dim).
    """
    return np.array([e.unwrapped._get_obs() for e in env.envs])


# --- Training Loops ---
def train_dynamics(dynamics, optimizer, replay_buffer, epochs, batch_size, device, val_ratio=0.1, writer=None, global_step=None):
    states, actions, next_states = replay_buffer
    num_samples = len(states)
    
    # 1. Shuffle & split into train and validation sets
    indices = np.random.permutation(num_samples)
    val_size = max(1, int(num_samples * val_ratio))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    train_states, train_actions, train_next_states = states[train_indices], actions[train_indices], next_states[train_indices]
    val_states, val_actions, val_next_states = states[val_indices], actions[val_indices], next_states[val_indices]
    
    train_targets = train_next_states - train_states
    val_targets = val_next_states - val_states
    
    # 2. Update normalization statistics ONLY on train split to prevent data leakage
    dynamics.state_mean.data = train_states.mean(dim=0, keepdim=True).to(device)
    dynamics.state_std.data = train_states.std(dim=0, keepdim=True).to(device)
    
    dynamics.action_mean.data = train_actions.mean(dim=0, keepdim=True).to(device)
    dynamics.action_std.data = train_actions.std(dim=0, keepdim=True).to(device)
    
    dynamics.delta_mean.data = train_targets.mean(dim=0, keepdim=True).to(device)
    dynamics.delta_std.data = train_targets.std(dim=0, keepdim=True).to(device)
    
    # 3. Normalize targets
    train_norm_targets = (train_targets.to(device) - dynamics.delta_mean) / (dynamics.delta_std + 1e-8)
    val_norm_targets = (val_targets.to(device) - dynamics.delta_mean) / (dynamics.delta_std + 1e-8)
    
    train_dataset = torch.utils.data.TensorDataset(train_states.to(device), train_actions.to(device), train_norm_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_states.to(device), val_actions.to(device), val_norm_targets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Training loop with validation tracking and best-weights restoration
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        dynamics.train()
        train_loss = 0.0
        for b_states, b_actions, b_norm_targets in train_loader:
            optimizer.zero_grad()
            norm_preds = dynamics(b_states, b_actions, return_normalized=True)
            loss = F.mse_loss(norm_preds, b_norm_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_states)
        train_loss /= len(train_dataset)
        
        # Evaluate validation loss
        dynamics.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_states, b_actions, b_norm_targets in val_loader:
                norm_preds = dynamics(b_states, b_actions, return_normalized=True)
                loss = F.mse_loss(norm_preds, b_norm_targets)
                val_loss += loss.item() * len(b_states)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in dynamics.state_dict().items()}
            
    # Restore best weights corresponding to minimum validation error
    if best_weights is not None:
        dynamics.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    print(f"Dynamics Training: Train MSE = {train_loss:.6f} | Best Val MSE = {best_val_loss:.6f}")
    if writer is not None and global_step is not None:
        writer.add_scalar("losses/dyn_train_mse", train_loss, global_step)
        writer.add_scalar("losses/dyn_val_mse", best_val_loss, global_step)
        
    return best_val_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"MBMF_{env_id}_{seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    env = init_gymnasium(env_id, capture_video=False, run_name=run_name, gamma=gamma, num_envs=num_envs, seed=seed)
    
    writer = SummaryWriter(f"runs/{run_name}")
    mb_writer = SummaryWriter(f"runs/{run_name}_Phase1_MB")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([
            f"|env_id|{env_id}|",
            f"|num_envs|{num_envs}|",
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
            f"|sequential_rollout_passes|{sequential_rollout_passes}|",
            f"|dyn_batch_size|{dyn_batch_size}|",
            f"|warmup_steps_per_env|{warmup_steps_per_env}|",
            f"|warmup_logstd|{warmup_logstd}|",
            f"|warmup_fit_rounds|{warmup_fit_rounds}|",
            f"|target_kl|{target_kl}|",
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
    
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    act_space = getattr(env, "single_action_space", env.action_space)
    state_dim = np.array(obs_space.shape).prod()
    action_dim = np.prod(act_space.shape)
    
    # Initialize Models
    dynamics_model = DynamicsModel(state_dim, action_dim).to(device)
    dyn_optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    
    agent = Agent(env).to(device)
    # eps=1e-5 to match ppo.py exactly (PyTorch's default is 1e-8).
    agent_optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    mpc = MPCController(env, dynamics_model, mpc_horizon, mpc_num_sequences, device)
    
    # Storage for Replay Buffer:
    # states_buf, actions_buf, next_states_buf store RAW physical states (bypassing NormalizeObservation) for Dynamics & MPC.
    # norm_states_buf stores normalized observations for Behavioral Cloning.
    states_buf, actions_buf, next_states_buf = [], [], []
    norm_states_buf = []

    # =================================================================================
    # Phase 1: Model-Based Pre-training (Train Dynamics & Gather MPC Data)
    # =================================================================================
    print("=== Phase 1: Model-Based Data Collection & Dynamics Training ===")
    
    # 1. Collect Initial Random Dataset (TF)
    print("Collecting Initial Random Dataset (D_RAND)...")
    obs, _ = env.reset(seed=seed)
    raw_obs = get_raw_obs_vector(env)
    steps_needed = max(1, mb_random_timesteps // num_envs)
    rand_ep_returns, rand_ep_lengths = [], []
    # Under NEXT_STEP autoreset, the step AFTER an episode ends ignores the submitted
    # action and returns a freshly reset observation. Storing that as a transition would
    # teach the dynamics model a physically impossible jump (s_final -> s_reset), so we
    # remember which envs just ended and skip them on the following step.
    prev_done = np.zeros(num_envs, dtype=bool)
    skipped = 0
    for t in range(steps_needed):
        actions = env.action_space.sample()  # Random actions for initial exploration
        next_obs, reward, terminated, truncated, infos = env.step(actions)
        raw_next_obs = get_raw_obs_vector(env)

        for i in range(num_envs):
            if prev_done[i]:
                skipped += 1
                continue
            states_buf.append(raw_obs[i])
            actions_buf.append(actions[i])
            next_states_buf.append(raw_next_obs[i])
            norm_states_buf.append(obs[i])
        prev_done = np.asarray(terminated) | np.asarray(truncated)

        obs = next_obs
        raw_obs = raw_next_obs

        if "_episode" in infos:
            for i, ep_done in enumerate(infos["_episode"]):
                if ep_done:
                    rand_ep_returns.append(infos["episode"]["r"][i])
                    rand_ep_lengths.append(infos["episode"]["l"][i])

    if len(rand_ep_returns) > 0:
        m_ret = float(np.mean(rand_ep_returns))
        m_len = float(np.mean(rand_ep_lengths))
        writer.add_scalar("charts/episodic_return", m_ret, mb_random_timesteps)
        writer.add_scalar("charts/episodic_length", m_len, mb_random_timesteps)
        mb_writer.add_scalar("charts/episodic_return", m_ret, mb_random_timesteps)
        mb_writer.add_scalar("charts/episodic_length", m_len, mb_random_timesteps)
            
    # D_RAND ends here. Recorded explicitly because the autoreset filter above means it
    # is slightly shorter than mb_random_timesteps; the 10/90 mix below indexes on it.
    d_rand_size = len(states_buf)
    print(f"D_RAND: {d_rand_size} usable transitions ({skipped} autoreset transitions skipped)")

    # 2. Train Dynamics on D_RAND
    print(f"Training Dynamics Model on D_RAND (TF) for {tf_dyn_epochs} epochs...")
    rb_rand = (torch.FloatTensor(np.array(states_buf)),
               torch.FloatTensor(np.array(actions_buf)),
               torch.FloatTensor(np.array(next_states_buf)))
    train_dynamics(dynamics_model, dyn_optimizer, rb_rand, tf_dyn_epochs, dyn_batch_size, device, writer=writer, global_step=mb_random_timesteps)

    # 3. DAgger MPC Aggregation Iterations (F)
    print("Starting MPC Aggregation Iterations (F)...")
    global_t = mb_random_timesteps
    # Full rollout depth, as specified by the MB-MF paper.
    steps_per_rollout = rollout_length
    for iter in range(aggregation_iters):
        print(f"--- Aggregation Iteration {iter+1}/{aggregation_iters} ---")
        iter_ep_returns, iter_ep_lengths = [], []
        for r in range(sequential_rollout_passes):
            # Unique seeds. `seed + iter + r` collided badly (iter=0,r=1 == iter=1,r=0),
            # which made Phase 1 re-collect the same initial states dozens of times.
            # A vector env reset(seed=s) seeds its sub-envs s, s+1, ..., s+num_envs-1,
            # so successive passes must be spaced by num_envs.
            pass_seed = seed + (iter * sequential_rollout_passes + r) * num_envs
            obs, _ = env.reset(seed=pass_seed)
            raw_obs = get_raw_obs_vector(env)
            prev_done = np.zeros(num_envs, dtype=bool)
            for step in range(steps_per_rollout):
                # One batched planning call for all environments at once.
                actions = mpc.get_actions(raw_obs)
                actions += np.random.normal(0, 0.005, size=actions.shape)
                actions = np.clip(actions, act_space.low, act_space.high)

                next_obs, reward, terminated, truncated, infos = env.step(actions)
                raw_next_obs = get_raw_obs_vector(env)

                for i in range(num_envs):
                    if prev_done[i]:
                        continue  # autoreset step: not a real transition
                    states_buf.append(raw_obs[i])
                    actions_buf.append(actions[i])
                    next_states_buf.append(raw_next_obs[i])
                    norm_states_buf.append(obs[i])
                prev_done = np.asarray(terminated) | np.asarray(truncated)

                obs = next_obs
                raw_obs = raw_next_obs
                global_t += num_envs

                if "_episode" in infos:
                    for i, ep_done in enumerate(infos["_episode"]):
                        if ep_done:
                            iter_ep_returns.append(infos["episode"]["r"][i])
                            iter_ep_lengths.append(infos["episode"]["l"][i])

        if len(iter_ep_returns) > 0:
            m_ret = float(np.mean(iter_ep_returns))
            m_len = float(np.mean(iter_ep_lengths))
            print(f"Phase 1 (Step {global_t}): mean_episodic_return={m_ret:.2f}")
            writer.add_scalar("charts/episodic_return", m_ret, global_t)
            writer.add_scalar("charts/episodic_length", m_len, global_t)
            mb_writer.add_scalar("charts/episodic_return", m_ret, global_t)
            mb_writer.add_scalar("charts/episodic_length", m_len, global_t)
        
        # 4. Train Dynamics on the paper's 10% random / 90% on-policy mixture
        d_rl_size = len(states_buf) - d_rand_size
        num_d_rand_samples = max(1, int(d_rl_size / 9))
        d_rand_indices = np.random.choice(d_rand_size, size=num_d_rand_samples, replace=True)
        d_rl_indices = np.arange(d_rand_size, len(states_buf))
        train_indices = np.concatenate([d_rand_indices, d_rl_indices])
        
        print(f"Training Dynamics Model on Aggregated Dataset (F) for {f_dyn_epochs} epochs...")
        rb_agg = (torch.FloatTensor(np.array(states_buf)[train_indices]), 
                  torch.FloatTensor(np.array(actions_buf)[train_indices]), 
                  torch.FloatTensor(np.array(next_states_buf)[train_indices]))
        train_dynamics(dynamics_model, dyn_optimizer, rb_agg, f_dyn_epochs, dyn_batch_size, device, writer=writer, global_step=global_t)


    # =================================================================================
    # Phase 2: Behavioral Cloning (Initialize Policy from MPC)
    # =================================================================================
    print("\n=== Phase 2: Behavioral Cloning (Initialize Policy) ===")
    print("Cloning MPC actions to initialize the Model-Free policy using DAgger...")
    
    # Seed BC with the most recent MPC data (the expert's best-informed behaviour).
    # Clamped so it can never reach back into the random D_RAND portion.
    init_steps = min(bc_initial_rollouts * rollout_length, len(norm_states_buf) - d_rand_size)
    bc_states = list(np.array(norm_states_buf)[-init_steps:])
    bc_actions = list(np.array(actions_buf)[-init_steps:])
    print(f"Behavioral cloning on {init_steps} MPC-labelled states")

    # Only the actor is being cloned, so only the actor gets an optimizer.
    cloning_optimizer = optim.Adam(agent.actor_mean.parameters(), lr=cloning_lr)
    
    for dagger_iter in range(bc_dagger_iters):
        print(f"--- Phase 2 DAgger Iteration {dagger_iter+1}/{bc_dagger_iters} ---")
        
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
        
        if bc_rollouts_per_iter > 0 and dagger_iter < bc_dagger_iters - 1:
            print(f"Running Actor for {bc_rollouts_per_iter} rollouts to collect DAgger queries...")
            agent.eval()
            dagger_ep_returns, dagger_ep_lengths = [], []
            for r in range(bc_rollouts_per_iter):
                # Spaced by num_envs for the same reason as Phase 1, and offset well past
                # the Phase 1 seed range so the two phases see different initial states.
                dagger_seed = (seed + 100000
                               + (dagger_iter * bc_rollouts_per_iter + r) * num_envs)
                obs, _ = env.reset(seed=dagger_seed)
                raw_obs = get_raw_obs_vector(env)
                for step in range(steps_per_rollout):
                    with torch.no_grad():
                        actor_action = agent.actor_mean(torch.FloatTensor(obs).to(device)).cpu().numpy()
                    
                    next_obs, reward, terminated, truncated, infos = env.step(actor_action)
                    raw_next_obs = get_raw_obs_vector(env)
                    global_t += num_envs
                    
                    if "_episode" in infos:
                        for i, ep_done in enumerate(infos["_episode"]):
                            if ep_done:
                                dagger_ep_returns.append(infos["episode"]["r"][i])
                                dagger_ep_lengths.append(infos["episode"]["l"][i])
                    
                    # DAgger: label the state the LEARNER visited with the EXPERT's action.
                    # Queried on raw_obs/obs, i.e. the state before this step, which is the
                    # state the actor actually chose its action in.
                    expert_actions = mpc.get_actions(raw_obs)
                    expert_actions += np.random.normal(0, 0.005, size=expert_actions.shape)
                    expert_actions = np.clip(expert_actions, act_space.low, act_space.high)
                    for i in range(num_envs):
                        bc_states.append(obs[i])
                        bc_actions.append(expert_actions[i])

                    obs = next_obs
                    raw_obs = raw_next_obs

            if len(dagger_ep_returns) > 0:
                m_ret = float(np.mean(dagger_ep_returns))
                m_len = float(np.mean(dagger_ep_lengths))
                print(f"Phase 2 (Step {global_t}): mean_episodic_return={m_ret:.2f}")
                writer.add_scalar("charts/episodic_return", m_ret, global_t)
                writer.add_scalar("charts/episodic_length", m_len, global_t)
                mb_writer.add_scalar("charts/episodic_return", m_ret, global_t)
                mb_writer.add_scalar("charts/episodic_length", m_len, global_t)
            agent.train()

    # =================================================================================
    # Phase 2.5: Critic Warm-Up
    # =================================================================================
    print("\n=== Phase 2.5: Critic Warm-Up ===")
    print("Freezing Actor and pre-training Critic on Actor's stochastic rollouts...")
    
    # Give the actor the exploration noise PPO will start from. See `warmup_logstd`:
    # a warm-started policy needs a NARROW distribution so the cloned mean survives.
    nn.init.constant_(agent.actor_logstd, warmup_logstd)

    for param in agent.actor_mean.parameters():
        param.requires_grad = False
    agent.actor_logstd.requires_grad = False

    agent.eval()
    obs, _ = env.reset(seed=seed + 200000)

    # Collect trajectory SEGMENTS rather than only completed episodes. Each segment
    # records how it ended, because that determines its value target at the far end:
    #   terminated       -> no future reward exists, bootstrap 0
    #   truncated/cutoff -> the process would have continued, bootstrap V(last state)
    # The previous version treated every segment end as terminal, which (with a 1000-step
    # TimeLimit and only 512 warm-up steps) meant no episode ever completed and every
    # single target was truncated at an arbitrary cutoff.
    segments = []
    ep_states = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    warmup_ep_returns, warmup_ep_lengths = [], []
    prev_done = np.zeros(num_envs, dtype=bool)

    for step in range(warmup_steps_per_env):
        for i in range(num_envs):
            if not prev_done[i]:  # skip the phantom autoreset step
                ep_states[i].append(obs[i])

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = np.clip(action.cpu().numpy(), act_space.low, act_space.high)

        next_obs, reward, terminated, truncated, infos = env.step(action)
        terminated, truncated = np.asarray(terminated), np.asarray(truncated)

        for i in range(num_envs):
            if not prev_done[i]:
                ep_rewards[i].append(float(reward[i]))

        done = terminated | truncated
        for i in range(num_envs):
            if done[i] and not prev_done[i] and len(ep_states[i]) > 0:
                segments.append({
                    "states": ep_states[i],
                    "rewards": ep_rewards[i],
                    # On a truncation the returned observation IS the true final state,
                    # so it is exactly what we need to bootstrap from.
                    "boot_obs": None if terminated[i] else np.array(next_obs[i]),
                })
                ep_states[i], ep_rewards[i] = [], []

        prev_done = done
        global_t += num_envs
        obs = next_obs

        if "_episode" in infos:
            for i, ep_done in enumerate(infos["_episode"]):
                if ep_done:
                    warmup_ep_returns.append(infos["episode"]["r"][i])
                    warmup_ep_lengths.append(infos["episode"]["l"][i])

    # Flush the unfinished tails. These were cut off by the data budget, not by the
    # environment, so they bootstrap from wherever they happened to stop.
    for i in range(num_envs):
        if len(ep_states[i]) > 0:
            segments.append({
                "states": ep_states[i],
                "rewards": ep_rewards[i],
                "boot_obs": np.array(obs[i]),
            })

    if len(warmup_ep_returns) > 0:
        m_ret = float(np.mean(warmup_ep_returns))
        m_len = float(np.mean(warmup_ep_lengths))
        print(f"Phase 2.5 (Step {global_t}): mean_episodic_return={m_ret:.2f}")
        writer.add_scalar("charts/episodic_return", m_ret, global_t)
        writer.add_scalar("charts/episodic_length", m_len, global_t)
        mb_writer.add_scalar("charts/episodic_return", m_ret, global_t)
        mb_writer.add_scalar("charts/episodic_length", m_len, global_t)

    agent.train()
    n_boot = sum(1 for s in segments if s["boot_obs"] is not None)
    print(f"Collected {len(segments)} segments "
          f"({sum(len(s['states']) for s in segments)} states, {n_boot} bootstrapped)")

    # 3. Train the Critic by fitted value iteration.
    # The bootstrap needs a critic, but the critic is what we are training. So targets are
    # recomputed each round using the improved critic; round 1 bootstraps from the (near
    # zero) freshly-initialized network and later rounds sharpen it.
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-3)

    for fit_round in range(warmup_fit_rounds):
        # Recompute targets with the current critic.
        boot_states = np.array([s["boot_obs"] for s in segments if s["boot_obs"] is not None])
        with torch.no_grad():
            if len(boot_states) > 0:
                boot_vals = agent.critic(
                    torch.FloatTensor(boot_states).to(device)
                ).squeeze(-1).cpu().numpy()
            else:
                boot_vals = np.array([])

        warmup_states, warmup_returns, b = [], [], 0
        for seg in segments:
            if seg["boot_obs"] is None:
                R = 0.0
            else:
                R = float(boot_vals[b]); b += 1
            rets = []
            for r in reversed(seg["rewards"]):
                R = r + gamma * R
                rets.insert(0, R)
            warmup_states.extend(seg["states"])
            warmup_returns.extend(rets)

        warmup_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.array(warmup_states)),
            torch.FloatTensor(np.array(warmup_returns)),
        )
        warmup_loader = torch.utils.data.DataLoader(warmup_dataset, batch_size=256, shuffle=True)

        for epoch in range(warmup_epochs):
            epoch_loss = 0
            for b_states, b_returns in warmup_loader:
                b_states, b_returns = b_states.to(device), b_returns.to(device)

                critic_optimizer.zero_grad()
                values = agent.critic(b_states).squeeze(-1)
                loss = F.mse_loss(values, b_returns)
                loss.backward()
                critic_optimizer.step()
                epoch_loss += loss.item()

        print(f"Critic fit round {fit_round+1}/{warmup_fit_rounds}, "
              f"final epoch loss: {epoch_loss/len(warmup_loader):.4f}, "
              f"target mean={np.mean(warmup_returns):.3f}")

    # 4. Unfreeze the Actor
    for param in agent.actor_mean.parameters():
        param.requires_grad = True
    agent.actor_logstd.requires_grad = True
    print("Actor unfrozen. Critic Warm-Up complete!")


    # =================================================================================
    # Phase 3: Model-Free Fine-Tuning (PPO)
    # =================================================================================
    print("\n=== Phase 3: Model-Free Fine-Tuning (PPO) ===")
    print("Starting Model-Free Fine-Tuning...")
    
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    # PPO only gets the budget the model-based phases did not already spend, so the
    # MB-MF arm and the pure-PPO baseline consume the same total number of env steps.
    remaining_timesteps = mf_total_timesteps - global_t
    num_iterations = max(1, remaining_timesteps // batch_size)
    print(f"Model-based phases used {global_t} steps; PPO gets the remaining {remaining_timesteps} "
          f"({num_iterations} iterations)")

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



# NOTE: an earlier version of this file concluded that model-based initialization "permanently
# handicaps" the agent. That conclusion was drawn while several bugs were active, any one of
# which would have produced the same symptom independently of the hypothesis:
#   - the actor was handed std = exp(0.5) = 1.65 at the PPO handoff, WIDER than PPO's own
#     from-scratch default, so >50% of actions saturated the clip bound and the cloned
#     behaviour was erased exactly when it was supposed to be exploited;
#   - the critic warm-up could never complete an episode, so 100% of its value targets were
#     Monte-Carlo returns truncated at an arbitrary cutoff (45% of them below 90% of true value);
#   - gamma differed from the PPO baseline on Swimmer (0.99 vs 0.9999);
#   - MPC rollouts were 1/num_envs of the paper's depth, so the expert data covered only the
#     first ~4% of an episode.
# The conclusion should be re-evaluated now that these are fixed.

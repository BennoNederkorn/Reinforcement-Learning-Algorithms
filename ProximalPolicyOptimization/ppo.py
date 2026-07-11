import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter



# General arguments
seed: int = 1
torch_deterministic: bool = True      # if True, torch.backends.cudnn.deterministic=False
cuda: bool = True                     # if True, cuda will be enabled
track: bool = False                   # if True, this experiment will be tracked with Weights and Biases
wandb_project_name: str = "PPO_tests" # the project name of WandB (Weights & Biases)
capture_video: bool = False           # if True, videos of the agent performances will be captured
save_model: bool = True               # if True, model will be saved into the runs/{run_name} folder

# Algorithm specific arguments
# The MuJoCo environment name
# The PPO paper used: "HalfCheetah-v5", "Hopper-v5", "InvertedDoublePendulum-v5", "InvertedPendulum-v5", "Reacher-v5", "Swimmer-v5", "Walker2d-v5"
# The mb-mf paper used: "Ant-v5", "HalfCheetah-v5", "Hopper-v5" and "Swimmer-v5"
# For the comparison I will use only "HalfCheetah-v5", "Hopper-v5" and "Swimmer-v5"
env_id: str = "HalfCheetah-v5" # S ∈ R23, A ∈ R6
env_id: str = "Hopper-v5"      # S ∈ R17, A ∈ R3
env_id: str = "Swimmer-v5"     # S ∈ R16, A ∈ R2

total_timesteps: int = 1000000   # total timesteps of the experiments
learning_rate: float = 3e-4      # learning rate of the optimizer
num_steps: int = 2048            # horizon: the number of steps the agent takes in each environment before it stops to learn
anneal_lr: bool = True           # If True, the optimizer's learning rate will be gradually decreased over the course of training for the policy and value networks
gamma: float = 0.99              # discount factor: determines how much the agent values future rewards compared to immediate rewards
gae_lambda: float = 0.95         # lambda for the general advantage estimation. It balance the bias-variance trade-off when estimating the advantage function A(s,a):  λ=0 (High Bias, Low Variance), λ=1 (Low Bias, High Variance)
num_minibatches: int = 32        # the number of mini-batches
update_epochs: int = 10          # the number (K) of epochs to update the policy
normalize_advantage: bool = True # If true the advantage is normalized. 
clip_coef: float = 0.2           # the surrogate clipping coefficient. 0.2 should be the best regarding the PPO paper.
entropy_coef: float = 0.0        # coefficient of the entropy controls the weight of the entropy bonus added to the training loss function
# the ppo paper used 0.1 for discrete action spaces, to support exploration and prevent the agent's policy from collapsing into a suboptimal strategy.
# the ppo paper used 0.0 for continuous action spaces, because the continuous Normal distribution's variance already regulates exploration
vf_coef: float = 0.5 # coefficient of the value function controls the relative importance of the value function loss 
max_grad_norm: float = 0.5       # the maximum norm for the gradient clipping controls
target_kl: float = None          # the targetKullback-Leibler Divergence threshold measures how much one probability distribution differs from another.

# to be filled in runtime
batch_size: int = 0     # = num_env * num_steps: The total number of (state, action, reward) experiences before the next gradient update is calculated.
minibatch_size: int = 0 # = batch_size // num_minibatches===the number of small updates per epoch.
num_iterations: int = 0 # The number of the "Collect Data → Update Policy" cycle.

# 
def init_gymnasium(env_id, capture_video, run_name, gamma):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()

        # Actor: predicts the mean of the next move for a given state. Input is the observation space.
        actor_fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 64)
        actor_fc2 = nn.Linear(64, 64)
        actor_fc3 = nn.Linear(64, np.prod(env.action_space.shape))

        # Initialize actor weights
        torch.nn.init.orthogonal_(actor_fc1.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(actor_fc1.bias, val=0.0)
        torch.nn.init.orthogonal_(actor_fc2.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(actor_fc2.bias, val=0.0)
        torch.nn.init.orthogonal_(actor_fc3.weight, gain=0.01)
        torch.nn.init.constant_(actor_fc3.bias, val=0.0)

        self.actor_mean = nn.Sequential(
            actor_fc1,
            nn.Tanh(),
            actor_fc2,
            nn.Tanh(),
            actor_fc3,
        )

        # Critic: predicts the value of being in a specific state. Input is the observation space. 
        critic_fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 64)
        critic_fc2 = nn.Linear(64, 64)
        critic_fc3 = nn.Linear(64, 1)

        # Initialize critic weights
        torch.nn.init.orthogonal_(critic_fc1.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(critic_fc1.bias, val=0.0)
        torch.nn.init.orthogonal_(critic_fc2.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(critic_fc2.bias, val=0.0)
        torch.nn.init.orthogonal_(critic_fc3.weight, gain=1.0)
        torch.nn.init.constant_(critic_fc3.bias, val=0.0)

        self.critic = nn.Sequential(
            critic_fc1,
            nn.Tanh(),
            critic_fc2,
            nn.Tanh(),
            critic_fc3,
        )

        # learnable parameter for the standard deviation: doesn't depend on the current state
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def rollout(env, agent, obs, actions, logprobs, rewards, dones, values, next_obs, next_done, num_steps, global_step, device, writer):
    for step in range(0, num_steps):
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # Unbatch the action before stepping
        next_obs, reward, terminations, truncations, infos = env.step(action.cpu().numpy()[0])
        next_done = terminations or truncations

        # Manually auto-reset the environment if the episode ended
        if next_done:
            next_obs, _ = env.reset()

        # Re-batch the outputs so PPO tensors don't break
        rewards[step] = torch.tensor([reward]).to(device).view(-1)
        next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
        next_done = torch.Tensor([next_done]).to(device)

        if "episode" in infos:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
            
    return next_obs, next_done, global_step


def compute_advantages(agent, next_obs, next_done, rewards, dones, values, num_steps, gamma, gae_lambda, device):
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    return advantages, returns


def update_policy(
    agent, optimizer, env, batch_size, minibatch_size, update_epochs,
    obs, logprobs, actions, advantages, returns, values,
    clip_coef, normalize_advantage, entropy_coef, vf_coef, max_grad_norm, target_kl
):
    # flatten the batch
    b_obs = obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + env.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    
    pg_loss = torch.tensor(0.0)
    v_loss = torch.tensor(0.0)
    entropy_loss = torch.tensor(0.0)
    old_approx_kl = torch.tensor(0.0)
    approx_kl = torch.tensor(0.0)

    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if normalize_advantage:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)

            # Clipping
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - entropy_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None and approx_kl > target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, np.mean(clipfracs), explained_var

def train_ppo(
    env, agent, optimizer, device, writer, num_steps, num_iterations, 
    batch_size, minibatch_size, update_epochs, learning_rate, anneal_lr, 
    gamma, gae_lambda, clip_coef, normalize_advantage, entropy_coef, 
    vf_coef, max_grad_norm, target_kl, save_model, run_name, seed,
    global_step=0
):
    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, 1) + env.observation_space.shape).to(device)
    actions = torch.zeros((num_steps, 1) + env.action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, 1)).to(device)
    rewards = torch.zeros((num_steps, 1)).to(device)
    dones = torch.zeros((num_steps, 1)).to(device)
    values = torch.zeros((num_steps, 1)).to(device)

    # gets the environment ready for step 1 of the training loop
    start_time = time.time()
    next_obs, _ = env.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
    next_done = torch.zeros(1).to(device)

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 1. Rollout / Data Collection
        next_obs, next_done, global_step = rollout(
            env, agent, obs, actions, logprobs, rewards, dones, values, 
            next_obs, next_done, num_steps, global_step, device, writer
        )

        # 2. Compute Advantages
        advantages, returns = compute_advantages(
            agent, next_obs, next_done, rewards, dones, values, 
            num_steps, gamma, gae_lambda, device
        )

        # 3. Update Policy and Value Network
        pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, mean_clipfrac, explained_var = update_policy(
            agent, optimizer, env, batch_size, minibatch_size, update_epochs,
            obs, logprobs, actions, advantages, returns, values,
            clip_coef, normalize_advantage, entropy_coef, vf_coef, max_grad_norm, target_kl
        )

        # 4. Logging
        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", mean_clipfrac, global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    if save_model:
        model_path = f"runs/{run_name}/model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")


if __name__ == "__main__":
    batch_size = int(num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size
    run_name = f"PPO_{env_id}_{seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([
            f"|seed|{seed}|",
            f"|torch_deterministic|{torch_deterministic}|",
            f"|track|{track}|",
            f"|capture_video|{capture_video}|",
            f"|save_model|{save_model}|",
            f"|env_id|{env_id}|",
            f"|total_timesteps|{total_timesteps}|",
            f"|learning_rate|{learning_rate}|",
            f"|num_steps|{num_steps}|",
            f"|anneal_lr|{anneal_lr}|",
            f"|gamma|{gamma}|",
            f"|gae_lambda|{gae_lambda}|",
            f"|num_minibatches|{num_minibatches}|",
            f"|update_epochs|{update_epochs}|",
            f"|normalize_advantage|{normalize_advantage}|",
            f"|clip_coef|{clip_coef}|",
            f"|entropy_coef|{entropy_coef}|",
            f"|vf_coef|{vf_coef}|",
            f"|max_grad_norm|{max_grad_norm}|",
            f"|target_kl|{target_kl}|",
            f"|batch_size|{batch_size}|",
            f"|minibatch_size|{minibatch_size}|",
            f"|num_iterations|{num_iterations}|"
        ]),
    )

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")

    # Environment setup
    env = init_gymnasium(env_id, capture_video, run_name, gamma)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # instantiates the neural network and moves it to GPU
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Call the training loop
    train_ppo(
        env=env, 
        agent=agent, 
        optimizer=optimizer, 
        device=device, 
        writer=writer, 
        num_steps=num_steps, 
        num_iterations=num_iterations, 
        batch_size=batch_size, 
        minibatch_size=minibatch_size, 
        update_epochs=update_epochs, 
        learning_rate=learning_rate, 
        anneal_lr=anneal_lr, 
        gamma=gamma, 
        gae_lambda=gae_lambda, 
        clip_coef=clip_coef, 
        normalize_advantage=normalize_advantage, 
        entropy_coef=entropy_coef, 
        vf_coef=vf_coef, 
        max_grad_norm=max_grad_norm, 
        target_kl=target_kl, 
        save_model=save_model, 
        run_name=run_name, 
        seed=seed
    )

    env.close()
    if writer:
        writer.close()
"""Proximal Policy Optimization (PPO) for continuous-control MuJoCo tasks.

PPO (Schulman et al., 2017, https://arxiv.org/abs/1707.06347) is an on-policy
actor-critic method. Each training iteration repeats three phases:

  1. ROLLOUT   Run the current policy for `num_steps` steps in `num_envs`
               parallel environments, storing every transition.
  2. ADVANTAGE Score each stored action with Generalized Advantage Estimation
               (GAE): "how much better was this action than what the critic
               expected on average?"
  3. UPDATE    Do several epochs of minibatch SGD on that fixed batch, using a
               *clipped* surrogate objective that prevents the policy from
               moving too far away from the one that collected the data.

Phase 3 is what makes PPO "proximal". A vanilla policy-gradient update may only
use each batch once, because after one gradient step the data is off-policy.
PPO reuses the batch for `update_epochs` passes and keeps the update honest with
an importance-sampling ratio r(θ) = π_θ(a|s) / π_θ_old(a|s) that is clipped to
[1-ε, 1+ε], so an update can never be driven arbitrarily far by a single
high-advantage sample.

This implementation follows the well-known CleanRL reference
(https://docs.cleanrl.dev/rl-algorithms/ppo/) and the "37 Implementation Details
of PPO" write-up, with two corrections that matter for the environments used
here (see `compute_advantages` and `update_policy`).

Run with:  python3 ProximalPolicyOptimization/ppo.py
Monitor with:  tensorboard --logdir runs
"""

import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Configuration
# =============================================================================
# These are module-level globals rather than argparse flags so the file stays
# readable as a single self-contained script. Edit them in place to run a
# different experiment.

# --- General ---
seed: int = 1
torch_deterministic: bool = True  # sets torch.backends.cudnn.deterministic
cuda: bool = True                 # use the GPU when one is available
capture_video: bool = False       # record videos of env 0 into videos/{run_name}
save_model: bool = True           # save final weights to runs/{run_name}/model.pt

# --- Environment ---
# The PPO paper used: HalfCheetah, Hopper, InvertedDoublePendulum,
#   InvertedPendulum, Reacher, Swimmer, Walker2d (all -v5 here).
# The MB-MF paper used: Ant, HalfCheetah, Hopper, Swimmer.
# This project compares the three below.
env_id: str = "HalfCheetah-v5"   # S ∈ R^17, A ∈ R^6
# env_id: str = "Hopper-v5"      # S ∈ R^11, A ∈ R^3
# env_id: str = "Swimmer-v5"     # S ∈ R^8,  A ∈ R^2

# --- Data collection ---
num_envs: int = 8                  # environments stepped in parallel
num_steps: int = 2048 // num_envs  # steps per env per iteration; batch stays 2048
total_timesteps: int = 1000000     # total env steps for the whole run

# --- Optimization ---
learning_rate: float = 3e-4    # Adam step size
anneal_lr: bool = True         # decay the LR linearly to 0 over the run
num_minibatches: int = 32      # minibatches per epoch => minibatch_size = 2048/32 = 64
update_epochs: int = 10        # passes over each collected batch
max_grad_norm: float = 0.5     # global gradient-norm clip

# --- PPO objective ---
gae_lambda: float = 0.95          # GAE bias/variance knob: 0 = low variance (TD), 1 = unbiased (MC)
clip_coef: float = 0.2           # ε in the clipped surrogate; 0.2 is the paper's value
normalize_advantage: bool = True  # standardize advantages within each minibatch
entropy_coef: float = 0.0         # entropy bonus. The PPO paper uses 0.0 for continuous
                                  # control, because the Gaussian's learned std already
                                  # controls exploration (it uses ~0.01 for discrete actions).
vf_coef: float = 0.5              # weight of the value loss in the total loss
clip_vloss: bool = True           # use the clipped value loss (CleanRL default)
target_kl: float = 0.015          # stop the update early if the policy has moved this far.
                                  # Set to None to always run all `update_epochs`.

# The discount factor sets the effective planning horizon, roughly 1/(1-gamma) steps.
# Swimmer needs special treatment: it never terminates, and its forward-swimming gait
# only pays off over hundreds of steps. With gamma=0.99 the agent only "sees" ~100
# steps ahead and plateaus near a return of 40; gamma=0.9999 reaches ~100.
gamma: float = 0.9999 if env_id.startswith("Swimmer") else 0.99


# =============================================================================
# Environment construction
# =============================================================================

def make_env(env_id, seed, idx, capture_video, run_name, gamma):
    """Build a *factory* for one environment instance.

    Vector environments need a list of zero-argument callables (not built
    environments), so this returns a closure ("thunk") instead of an env.

    The wrapper order matters and follows the standard PPO/MuJoCo recipe:
    RecordEpisodeStatistics sits close to the raw env so that the returns it
    logs are the true, unnormalized scores you can compare against published
    numbers, rather than the rescaled rewards the agent actually trains on.
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)      # flatten dict/nested observations
        env = gym.wrappers.RecordEpisodeStatistics(env)  # log RAW episode return/length
        env = gym.wrappers.ClipAction(env)               # clip actions into the valid Box

        # Neural nets train best on roughly zero-mean, unit-variance inputs. These two
        # wrappers keep a running mean/std of observations and standardize them, then
        # clip outliers so a single strange state cannot blow up the forward pass.
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space
        )

        # NormalizeReward divides rewards by the running std of the *discounted return*,
        # bringing them to ~unit scale no matter their raw magnitude. That keeps the value
        # loss on a consistent scale across environments, which matters because vf_coef and
        # the value-clipping range are absolute numbers. Applied to every environment so the
        # HalfCheetah / Hopper / Swimmer comparison uses an identical pipeline.
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        env.action_space.seed(seed + idx)
        return env

    return thunk


def init_gymnasium(env_id, capture_video, run_name, gamma, num_envs=1, seed=1):
    """Create `num_envs` independent environments stepped together in lockstep.

    SyncVectorEnv steps them sequentially in one process. Collecting from several
    environments at once decorrelates consecutive samples in a batch, which makes
    the gradient estimate less noisy than a single long trajectory would.
    """
    env_fns = [make_env(env_id, seed, i, capture_video, run_name, gamma) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)


def _space_shapes(env):
    """Return (observation_shape, action_shape) for one environment.

    Vector envs expose `single_*_space` for the per-env spaces; plain envs only
    have `observation_space` / `action_space`. Supporting both lets these helpers
    be reused (e.g. by the MB-MF script) without assuming a vectorized env.
    """
    obs_space = getattr(env, "single_observation_space", getattr(env, "observation_space", None))
    act_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    return obs_space.shape, act_space.shape


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight init, the standard choice for PPO.

    Orthogonal matrices preserve the norm of the signal passing through them, which
    keeps activations and gradients well-scaled in deep stacks. The `std` gain is
    tuned per layer type:
      sqrt(2)  hidden layers with tanh/ReLU, compensating for the activation
      0.01     the actor's output layer, so the initial policy is nearly identical
               across actions (near-uniform exploration instead of an arbitrary bias)
      1.0      the critic's output layer, since values are unbounded regressions
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Agent: separate actor and critic networks
# =============================================================================

class Agent(nn.Module):
    """Actor-critic for continuous actions.

    The actor outputs the *mean* of a diagonal Gaussian over actions. The
    log-standard-deviation is a free parameter that does NOT depend on the state
    ("state-independent exploration"): it is one learnable number per action
    dimension. This is the standard PPO/TRPO choice for MuJoCo — it lets the agent
    anneal its own exploration globally as training converges, and it is far easier
    to optimize than a state-conditioned std.

    Actor and critic are kept as two separate networks (no shared trunk). Sharing
    would force one set of features to serve two objectives whose gradients differ
    in scale, which is a common source of instability on MuJoCo tasks.
    """

    def __init__(self, env):
        super().__init__()
        obs_shape, act_shape = _space_shapes(env)
        state_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(act_shape))

        # Actor: state -> mean of the action distribution
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, action_dim), std=0.01),
        )

        # Critic: state -> scalar value estimate V(s)
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

        # log(std) rather than std directly, so exp() keeps it positive without
        # constraints. Initialized to 0 => std = exp(0) = 1.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """V(s) for a batch of states. Shape: (batch, 1)."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Sample an action (or evaluate a given one) and estimate its state value.

        Passing `action=None` (rollout) samples a fresh action. Passing a stored
        action (update) re-evaluates it under the *current* policy parameters —
        that is how the importance ratio r(θ) in the PPO loss is obtained.

        Returns (action, log_prob, entropy, value). log_prob and entropy are summed
        over action dimensions because the Gaussian is diagonal, so the joint
        density factorizes and log-probabilities add.
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# =============================================================================
# Phase 1: rollout — collect a batch of on-policy experience
# =============================================================================

def rollout(envs, agent, obs, actions, logprobs, rewards, dones, truncs, values,
            next_obs, next_done, next_trunc, num_steps, global_step, device, writer):
    """Step the environments `num_steps` times, filling the storage tensors.

    All storage tensors have shape (num_steps, num_envs, ...) and are allocated
    once by `train_ppo` and overwritten every iteration.

    IMPORTANT — index alignment. `obs[t]`, `dones[t]` and `truncs[t]` describe the
    state *before* taking `actions[t]`, while `rewards[t]` is the reward received
    *after* it. So `dones[t] == 1` means "the episode ended just before index t",
    not "it ended at index t". `compute_advantages` and `update_policy` both rely
    on this convention.

    `next_obs` / `next_done` / `next_trunc` are carried in and out across
    iterations so trajectories continue seamlessly from one batch to the next.
    """
    num_envs = envs.num_envs
    ep_returns, ep_lengths = [], []

    for step in range(0, num_steps):
        global_step += num_envs
        obs[step] = next_obs
        dones[step] = next_done
        truncs[step] = next_trunc

        # No gradients here: this is data collection. The policy that generated
        # these actions is frozen and becomes "π_old" for the coming update.
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())

        # Two very different reasons an episode can end, kept separate on purpose:
        #   terminated - the MDP genuinely reached a terminal state (Hopper fell over)
        #   truncated  - we hit the TimeLimit (1000 steps) and simply stopped looking
        # `done` merges them for bookkeeping; `compute_advantages` needs them apart.
        next_done_np = terminations | truncations

        rewards[step] = torch.tensor(reward_np, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
        next_done = torch.tensor(next_done_np, dtype=torch.float32).to(device)
        next_trunc = torch.tensor(truncations, dtype=torch.float32).to(device)

        # RecordEpisodeStatistics reports finished episodes through the info dict.
        # The vector env aggregates the sub-environment infos into arrays plus a
        # boolean mask under the "_episode" key marking which entries are valid.
        if "_episode" in infos:
            for i, ep_done in enumerate(infos["_episode"]):
                if ep_done:
                    ep_returns.append(infos["episode"]["r"][i])
                    ep_lengths.append(infos["episode"]["l"][i])

    # Average over every episode that finished in this batch. With 8 envs and
    # 1000-step episodes only a couple finish per iteration, so a single episode
    # would be a very noisy progress signal.
    if len(ep_returns) > 0 and writer is not None:
        mean_return = float(np.mean(ep_returns))
        mean_length = float(np.mean(ep_lengths))
        print(f"global_step={global_step}, episodic_return={mean_return:.2f}")
        writer.add_scalar("charts/episodic_return", mean_return, global_step)
        writer.add_scalar("charts/episodic_length", mean_length, global_step)

    return next_obs, next_done, next_trunc, global_step


# =============================================================================
# Phase 2: Generalized Advantage Estimation
# =============================================================================

def compute_advantages(agent, next_obs, next_done, next_trunc,
                       rewards, dones, truncs, values, num_steps, gamma, gae_lambda, device):
    """Score every stored action with GAE (Schulman et al., https://arxiv.org/abs/1506.02438).

    The one-step TD residual measures how much better a step turned out than the
    critic predicted:

        δ_t = r_t + γ·V(s_{t+1}) − V(s_t)

    GAE is the exponentially-weighted sum of those residuals, computed efficiently
    by a single backwards recursion:

        A_t = δ_t + (γ·λ)·A_{t+1}

    λ interpolates between low-variance/high-bias (λ=0, plain TD) and
    high-variance/low-bias (λ=1, Monte-Carlo). λ=0.95 is the usual compromise.

    The value targets ("returns") are then A_t + V(s_t), i.e. the critic's own
    prediction corrected by the advantage it failed to anticipate.

    TERMINATION vs TRUNCATION. This is the subtle part. At an episode boundary we
    must decide whether future value still exists:

      - terminated: the MDP really ended, all future reward is 0, so do NOT
        bootstrap — drop the γ·V(s_{t+1}) term.
      - truncated: the episode was cut off by the TimeLimit, but the underlying
        process would have continued. We MUST still bootstrap γ·V(final_obs),
        otherwise the value target is pushed toward 0 at every horizon.

    Treating truncation as termination is a classic and costly bug here, because
    HalfCheetah and Swimmer *never* terminate — every one of their episodes ends by
    truncation, so the bias would apply to every episode.

    Where does V(final_obs) come from? Gymnasium's vector envs use NEXT_STEP
    autoreset: on the step after an episode ends, the env ignores the submitted
    action, resets, and returns reward 0. Because of the index alignment described
    in `rollout`, `obs[t+1]` at such a boundary holds the true final observation,
    so `values[t+1]` is exactly the V(final_obs) we need.

    The GAE recursion itself is cut at *any* boundary (termination or truncation),
    since A_{t+1} then belongs to a different episode.
    """
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(-1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # Past the end of the buffer: use the state the rollout stopped at.
                done_tp1, trunc_tp1, value_tp1 = next_done, next_trunc, next_value
            else:
                done_tp1, trunc_tp1, value_tp1 = dones[t + 1], truncs[t + 1], values[t + 1]

            # Bootstrap unless the episode genuinely terminated (done AND not truncated).
            terminated_tp1 = done_tp1 * (1.0 - trunc_tp1)
            delta = rewards[t] + gamma * value_tp1 * (1.0 - terminated_tp1) - values[t]

            # Stop the recursion at any episode boundary, truncation included.
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * (1.0 - done_tp1) * lastgaelam

        returns = advantages + values

    return advantages, returns


# =============================================================================
# Phase 3: the PPO update
# =============================================================================

def update_policy(agent, optimizer, envs, minibatch_size, update_epochs,
                  obs, logprobs, actions, advantages, returns, values, dones,
                  clip_coef, normalize_advantage, entropy_coef, vf_coef, clip_vloss,
                  max_grad_norm, target_kl):
    """Run several epochs of minibatch SGD on the collected batch.

    The storage tensors are (num_steps, num_envs, ...); everything is flattened to
    (batch_size, ...) because the update treats the batch as an unordered set of
    independent transitions (the temporal structure was already consumed by GAE).
    """
    obs_shape, act_shape = _space_shapes(envs)
    b_obs = obs.reshape((-1,) + obs_shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + act_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_dones = dones.reshape(-1)

    # Drop "phantom" transitions. Under NEXT_STEP autoreset, b_dones[i] == 1 marks
    # an index where the environment had already finished its episode: obs[i] is the
    # stale final observation, and the action we sampled there was silently discarded
    # by the env (which reset instead) and earned a reward of 0. Training on it would
    # teach the policy from a transition that never actually happened.
    b_inds = np.where(b_dones.cpu().numpy() == 0)[0]

    # Diagnostics. Pre-set so the returns are well-defined even if the epoch loop
    # exits immediately via the target_kl break.
    clipfracs = []
    pg_loss = torch.tensor(0.0)
    v_loss = torch.tensor(0.0)
    entropy_loss = torch.tensor(0.0)
    old_approx_kl = torch.tensor(0.0)
    approx_kl = torch.tensor(0.0)

    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)  # fresh minibatch partition every epoch

        for start in range(0, len(b_inds), minibatch_size):
            mb_inds = b_inds[start:start + minibatch_size]

            # Re-evaluate the STORED actions under the CURRENT policy. The gap between
            # newlogprob and the stored b_logprobs is what the clipping acts on.
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()  # r(θ) = π_θ(a|s) / π_θ_old(a|s)

            with torch.no_grad():
                # How far has the policy drifted from the one that collected the data?
                # approx_kl is Schulman's low-variance, always-non-negative KL estimator
                # (http://joschu.net/blog/kl-approx.html); old_approx_kl is the naive one.
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                # Fraction of samples where clipping was active — a useful health check.
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if normalize_advantage:
                # Standardizing per minibatch makes the gradient scale independent of the
                # reward scale, so one learning rate works across different environments.
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # --- Clipped policy objective ---
            # We MINIMIZE the negative surrogate, hence the leading minus signs, and take
            # the MAX of the two terms (a pessimistic lower bound on the objective). The
            # effect: improving an action is only rewarded while the ratio stays inside
            # [1-ε, 1+ε]; beyond that the gradient vanishes and the policy stops chasing it.
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # --- Value loss ---
            newvalue = newvalue.view(-1)
            if clip_vloss:
                # Same pessimistic trick for the critic: penalize whichever is worse, the
                # raw error or the error after limiting how far V may move in one update.
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # --- Total loss ---
            # Entropy is SUBTRACTED: maximizing entropy discourages premature collapse to
            # a deterministic policy. With entropy_coef = 0.0 this term does nothing.
            entropy_loss = entropy.mean()
            loss = pg_loss - entropy_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            # Rescale the whole gradient if its norm exceeds the threshold, so a single
            # bad batch cannot take a huge step.
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        # Early stopping: if the policy has already moved further than target_kl, the
        # remaining epochs would be optimizing on badly off-policy data. Note this
        # inspects only the LAST minibatch's KL, which is the usual (noisy) shortcut.
        if target_kl is not None and approx_kl > target_kl:
            break

    # explained_variance ≈ 1 means the critic explains the returns well; ≈ 0 means it is
    # no better than predicting the mean. The single most useful signal that the critic
    # is healthy. (Computed over the full batch including phantom steps, so it is a
    # slight underestimate — a logging detail only, it does not affect training.)
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, np.mean(clipfracs), explained_var


# =============================================================================
# Training loop
# =============================================================================

def train_ppo(env, agent, optimizer, device, writer, num_steps, num_iterations,
              batch_size, minibatch_size, update_epochs, learning_rate, anneal_lr,
              gamma, gae_lambda, clip_coef, normalize_advantage, entropy_coef,
              vf_coef, max_grad_norm, target_kl, save_model, run_name, seed,
              global_step=0, clip_vloss=True):
    """Repeat rollout -> advantages -> update for `num_iterations` iterations.

    `global_step` is a parameter so training can be resumed or continued from an
    earlier phase; the MB-MF script uses this to hand a pre-trained agent over to
    PPO with the step counter already advanced. `batch_size` is accepted for the
    same interface-compatibility reason.
    """
    num_envs = getattr(env, "num_envs", 1)
    obs_shape, act_shape = _space_shapes(env)

    # Allocated once and reused every iteration — PPO is on-policy, so last
    # iteration's data is discarded rather than stored in a replay buffer.
    obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + act_shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    truncs = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    start_time = time.time()

    # Reset once, at the very start. From then on the vector env auto-resets, and
    # these three variables carry the trajectory across iteration boundaries.
    next_obs, _ = env.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)
    next_trunc = torch.zeros(num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        # Linearly decay the LR to 0. Large steps early to explore, small steps late
        # to settle — a standard and reliably helpful PPO detail.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * learning_rate

        next_obs, next_done, next_trunc, global_step = rollout(
            env, agent, obs, actions, logprobs, rewards, dones, truncs, values,
            next_obs, next_done, next_trunc, num_steps, global_step, device, writer
        )

        advantages, returns = compute_advantages(
            agent, next_obs, next_done, next_trunc, rewards, dones, truncs, values,
            num_steps, gamma, gae_lambda, device
        )

        pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, mean_clipfrac, explained_var = update_policy(
            agent, optimizer, env, minibatch_size, update_epochs,
            obs, logprobs, actions, advantages, returns, values, dones,
            clip_coef, normalize_advantage, entropy_coef, vf_coef, clip_vloss,
            max_grad_norm, target_kl
        )

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

    if save_model:
        model_path = f"runs/{run_name}/model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # batch_size is the data collected per iteration; minibatch_size is one SGD step.
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

    run_name = f"PPO_{env_id}_{seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Record the full configuration in TensorBoard so a run stays reproducible
    # even after the file has been edited.
    hyperparameters = {
        "seed": seed, "torch_deterministic": torch_deterministic, "capture_video": capture_video,
        "save_model": save_model, "env_id": env_id, "num_envs": num_envs,
        "total_timesteps": total_timesteps, "learning_rate": learning_rate,
        "num_steps": num_steps, "anneal_lr": anneal_lr, "gamma": gamma,
        "gae_lambda": gae_lambda, "num_minibatches": num_minibatches,
        "update_epochs": update_epochs, "normalize_advantage": normalize_advantage,
        "clip_coef": clip_coef, "entropy_coef": entropy_coef, "vf_coef": vf_coef,
        "clip_vloss": clip_vloss, "max_grad_norm": max_grad_norm, "target_kl": target_kl,
        "batch_size": batch_size, "minibatch_size": minibatch_size,
        "num_iterations": num_iterations,
    }
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in hyperparameters.items()),
    )

    # Seed every source of randomness so a run can be reproduced exactly.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"device: {device}")

    env = init_gymnasium(env_id, capture_video, run_name, gamma, num_envs=num_envs, seed=seed)
    # The Gaussian actor only makes sense for continuous (Box) action spaces; a
    # discrete environment would need a Categorical policy instead.
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(env).to(device)
    # eps=1e-5 instead of PyTorch's 1e-8: another small but well-established PPO detail.
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

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
        clip_vloss=clip_vloss,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        save_model=save_model,
        run_name=run_name,
        seed=seed,
    )

    env.close()
    writer.close()

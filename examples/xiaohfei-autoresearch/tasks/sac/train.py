import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium.wrappers.vector import TransformReward
from jaxtyping import Float
from loguru import logger
from torch import Tensor, nn
from torch.optim import Adam
from tqdm.auto import tqdm  # Import tqdm for progress bar


################################################################################
# Reproducibility
################################################################################
def seed_everything(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed all RNGs that affect training so a run is reproducible.

    Covers Python's ``random``, NumPy's global RNG, and PyTorch (CPU + all CUDA
    devices). Also sets ``PYTHONHASHSEED`` so hash-based ordering is stable.

    Gym env streams (``env.reset(seed=...)``, ``env.action_space.seed(...)``)
    are seeded separately at their call sites -- they don't draw from these RNGs.

    Args:
        seed: The base seed.
        deterministic_cudnn: When True, force cuDNN into deterministic mode
            (``deterministic=True``, ``benchmark=False``). Reproducible but can
            be slightly slower; set False to trade determinism for speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


################################################################################
# Model definitions
################################################################################
class ActionValueNetwork(nn.Module):
    """The action-value network Q(s, a)"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        self.dim_obs = obs_dim
        self.dim_act = action_dim

        self.layers = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, s: Float[Tensor, "B dim_obs"], a: Float[Tensor, "B dim_act"]
    ) -> Float[Tensor, "B 1"]:
        x = torch.cat([s, a], dim=-1)
        x = self.layers(x)
        return x


class OutputScaler(nn.Module):
    """Given the unbounded output of linear layers, apply tanh to squash it to (-1, 1),
    and then scale it to the desired (low, high) range.
    Note, all elements of low & high should be finite.
    """

    def __init__(
        self,
        output_dim: int,
        output_low: Float[Tensor, "output_dim"],
        output_high: Float[Tensor, "output_dim"],
        eps: float = 1e-6,
    ):
        super().__init__()

        assert isinstance(output_low, Tensor)
        assert isinstance(output_high, Tensor)
        assert output_low.shape == (output_dim,)
        assert output_high.shape == (output_dim,)

        assert all(torch.isfinite(output_low))
        assert all(torch.isfinite(output_high))

        assert all(output_low < output_high)

        self.low = nn.Buffer(output_low)
        self.high = nn.Buffer(output_high)
        self.eps = eps

    def forward(
        self, x: Float[Tensor, "B output_dim"]
    ) -> tuple[Float[Tensor, "B output_dim"], Float[Tensor, "B 1"]]:
        y = self.low + (0.5 * (nn.functional.tanh(x) + 1.0) * (self.high - self.low))

        # Compute the log determinant of the Jacobian for the squashing transform:
        # dy/dx = 0.5 * (1 - tanh^2(x)) * (high - low)
        dy_dx = 0.5 * (1.0 - torch.tanh(x).pow(2)) * (self.high - self.low)

        # We add 1e-6 for numerical stability to avoid log(0)
        # Technically, we need to compute det(dy_dx):
        # 1/ Under the assumption that dy_dx is a diagonal matrix, det(dy_dx) is then the prod of the diagonal elements,
        # 2/ And then log(det(dy_dx)) = log(prod(dy_dx's elements))=sum(log(dy_dx's elements))
        log_det_jacobian = torch.sum(
            torch.log(dy_dx + self.eps), dim=-1, keepdim=True
        )  # (B, 1)
        assert log_det_jacobian.ndim == 2
        assert log_det_jacobian.shape[-1] == 1

        return y, log_det_jacobian


@dataclass
class PolicyNetworkOutput:
    action: Float[Tensor, "B dim_act"]
    """ Sampled action in the actual action space.
  """
    z: Float[Tensor, "B dim_act"]
    """ Standard Gaussian random variables corresponding to the sampled actions.
  """
    log_prob: Float[Tensor, "B 1"]
    """ log probability of the sampled actions under the policy.
  """
    entropy: Float[Tensor, "B 1"]
    r""" entropy of the policy \pi(A_t | S_t = s_t), that is, the entropy of the
  distribution over actions conditioned on the state.
  """


class PolicyNetwork(nn.Module):
    r"""The policy network \pi(A_t | S_t)"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: Float[Tensor, "dim_act"],
        action_high: Float[Tensor, "dim_act"],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dim_obs = obs_dim
        self.dim_act = action_dim

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_mean = nn.Linear(hidden_dim, action_dim)
        self.head_logstd = nn.Linear(hidden_dim, action_dim)

        self.output_scaler = OutputScaler(action_dim, action_low, action_high)

    def sample_deterministic(
        self, s: Float[Tensor, "B dim_obs"]
    ) -> Float[Tensor, "B dim_act"]:
        feat = self.layers(s)
        mean = self.head_mean(feat)
        y, _ = self.output_scaler(mean)
        return y

    def forward(self, s: Float[Tensor, "B dim_obs"]) -> PolicyNetworkOutput:
        feat = self.layers(s)

        mean = self.head_mean(feat)

        logstd = self.head_logstd(feat)
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)

        z = torch.randn_like(mean)

        x = mean + std * z  # (B, dim_act)
        y, log_det_jacobian = self.output_scaler(x)

        # Compute log probability of the base Gaussian sample x
        normal_dist = torch.distributions.Normal(mean, std)
        log_prob_x = torch.sum(normal_dist.log_prob(x), dim=-1, keepdim=True)  # (B, 1)
        assert log_prob_x.ndim == 2 and log_prob_x.shape[-1] == 1

        # p(y) = p(x) * |dy/dx|^-1
        # => log p(y) = log p(x) - log |dy/dx|
        log_prob_y = log_prob_x - log_det_jacobian  # (B, 1)
        assert log_prob_y.ndim == 2 and log_prob_y.shape[-1] == 1

        # (1 sample) Monte Carlo estimation of the entropy -- however, using closed-form + one-sample MC
        # should have lower variance
        # entropy_y = -log_prob_y

        # for multivariate (diagonal) Gaussian distribution,
        # the entropy is proportional to log(det(covariance_matrix))
        # since the covariance_matrix is diagonal, det(covariance_matrix)=prod var_i
        # and as such
        # entropy ~ log(det(cov)) = log(prod var_i)
        #                         = sum(log(var_i))
        #                         = sum(log(std_i^2))
        #                         ~ sum(log(std_i))
        # That is: entropy ~ sum(logstd)
        entropy_x = torch.sum(logstd, dim=-1, keepdim=True)  # (B, 1)
        assert entropy_x.ndim == 2 and entropy_x.shape[-1] == 1

        # However, since we transform x to y (y is our real sample),
        # we need to compute the entropy of y instead of x
        # h(Y) = h(X) + E[log|det(f'(X))|] where Y = f(X).
        # Our log_det_jacobian = log(det(df(x)/dx)) is a one-sample MC approximation
        # of the expectation term above.
        # Therefore we can use the closed-form solution of h(X) along with a MC
        # estimation of E[log ...].
        entropy_y = entropy_x + log_det_jacobian  # (B, 1)

        return PolicyNetworkOutput(
            action=y, z=z, log_prob=log_prob_y, entropy=entropy_y
        )


class ValueNetwork(nn.Module):
    """The (state) value network V(s)"""

    def __init__(self, obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        self.dim_obs = obs_dim
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: Float[Tensor, "B dim_obs"]) -> Float[Tensor, "B 1"]:
        v = self.layers(s)
        return v


################################################################################
# Training loop
################################################################################


def create_vectorized_env(
    env_name: str,
    max_episode_steps: int,
    num_envs: int,
    reward_scale: float,
    record_every_n_episodes: int = -1,
    record_video_folder: str = "./recordings",
) -> TransformReward:
    def make_env(env_id: int):
        def thunk():
            # this is the so called "thunk" trick ...
            _env = gym.make(
                env_name, max_episode_steps=max_episode_steps, render_mode="rgb_array"
            )
            if record_every_n_episodes > 0 and env_id == 0:
                _env = gym.wrappers.RecordVideo(
                    _env,
                    video_folder=record_video_folder,
                    episode_trigger=lambda episode_id: (
                        episode_id % record_every_n_episodes == 0
                    ),
                    # video_length=120  # number of frames
                )
                # _env.metadata["render_fps"] = 12
                logger.info(
                    f"recording @ every {record_every_n_episodes} episodes to dir: {record_video_folder}"
                )
            return _env

        return thunk

    envs = gym.vector.AsyncVectorEnv([make_env(env_id=i) for i in range(num_envs)])
    # envs = NormalizeObservation(envs)
    envs = TransformReward(envs, lambda reward: reward * reward_scale)

    return envs


class VectorReplayBuffer:
    def __init__(
        self, capacity: int, num_envs: int, obs_dim: int, act_dim: int, device: str
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device

        # Pre-allocate memory
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        B = self.num_envs
        if self.ptr + B <= self.capacity:
            self.obs[self.ptr : self.ptr + B] = obs
            self.actions[self.ptr : self.ptr + B] = action
            self.rewards[self.ptr : self.ptr + B] = reward
            self.next_obs[self.ptr : self.ptr + B] = next_obs
            self.dones[self.ptr : self.ptr + B] = done
            self.ptr = (self.ptr + B) % self.capacity
            self.size = min(self.size + B, self.capacity)
        else:
            # Wrap around: split into two chunks
            chunk1 = self.capacity - self.ptr
            chunk2 = B - chunk1

            self.obs[self.ptr : self.capacity] = obs[:chunk1]
            self.actions[self.ptr : self.capacity] = action[:chunk1]
            self.rewards[self.ptr : self.capacity] = reward[:chunk1]
            self.next_obs[self.ptr : self.capacity] = next_obs[:chunk1]
            self.dones[self.ptr : self.capacity] = done[:chunk1]

            self.obs[0:chunk2] = obs[chunk1:]
            self.actions[0:chunk2] = action[chunk1:]
            self.rewards[0:chunk2] = reward[chunk1:]
            self.next_obs[0:chunk2] = next_obs[chunk1:]
            self.dones[0:chunk2] = done[chunk1:]

            self.ptr = chunk2
            self.size = self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.next_obs[idxs], device=self.device),
            torch.as_tensor(self.dones[idxs], device=self.device),
        )

    def save(self, path: str):
        np.savez_compressed(
            path,
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            next_obs=self.next_obs,
            dones=self.dones,
            ptr=self.ptr,
            size=self.size,
        )
        logger.info(f"Replay buffer saved to {path}")

    def load(self, path: str):
        data = np.load(path)
        self.obs = data["obs"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_obs = data["next_obs"]
        self.dones = data["dones"]
        self.ptr = data["ptr"].item()
        self.size = data["size"].item()
        logger.info(f"Replay buffer loaded from {path}")


@dataclass
class EvaluationResult:
    average_cumulative_reward: float
    """ The average of cumulative reward -- the rewards are first cumulated over each trajectory 
    which are then averaged across the batch.
    """

    entropy: float
    """
    Monte Carlo estimation of the entropy.
    """


def evaluate_policy(
    eval_env: gym.vector.VectorEnv,
    policy: PolicyNetwork,
    T: int,
    device: str,
    seed: int | None = None,
) -> EvaluationResult:
    """Evaluate the policy on the given environment.

    The caller needs to
    1/ set the policy network to eval mode (and then set it back to train), and
    2/ scale the reward to the original scale.

    Pass ``seed`` to make evaluation deterministic: the eval env is reset to the
    same initial states on every call, so the returned reward reflects policy
    changes rather than a shifting evaluation distribution.
    """
    obs, info = eval_env.reset(seed=seed)

    cumulative_reward = []
    entropy = []

    for _t in range(T):
        with torch.no_grad():
            # act = policy.sample_deterministic(torch.as_tensor(obs, dtype=torch.float32, device=device))
            sample = policy(torch.as_tensor(obs, dtype=torch.float32, device=device))

        action = sample.action.cpu().numpy()
        next_obs, reward, terminated, truncated, info = eval_env.step(action)

        cumulative_reward.append(reward)
        # 1/ entropy is not cumulative
        # 2/ sample.entropy is actually a sample of the entropy
        # so to compute the real entropy, we do Monte Carlo integration, that is,
        # we need to take average of all the entropy samples
        entropy.append(sample.entropy.cpu().numpy())

        obs = next_obs

    mean_cumulative_reward = np.mean(
        np.sum(cumulative_reward)  # sum over timesteps
    )  # mean over batches
    entropy = np.mean(entropy)

    return EvaluationResult(
        average_cumulative_reward=mean_cumulative_reward.item(), entropy=entropy.item()
    )


SUPPORTED_ENVS = [
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
    "Ant-v5",
    "Humanoid-v5",
]


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ########################################
    # reproducibility options
    ########################################
    seed: int = 0
    """ Base seed for all RNGs (Python, NumPy, PyTorch, and gym env streams). """
    deterministic_cudnn: bool = True
    """ Force cuDNN into deterministic mode. Reproducible but can be slightly
    slower on GPU; set False to trade determinism for speed. """

    ########################################
    # environment-related options
    ########################################
    env_name: str = "HalfCheetah-v5"
    max_episode_steps: int = (
        1_000  # -1: no limit; standard across MuJoCo locomotion tasks
    )
    num_episodes: int = 1_000
    num_envs: int = 8
    reward_scale: float = 1.0
    record_every_n_episodes: int = -1  # -1 to disable recording
    max_timesteps_used: int = 1_000_000  # limit training to 1M timesteps
    learning_starts_at_n_timesteps: int = 1_000

    ########################################
    # learning-related options
    ########################################
    alpha: float = 0.2
    """ coefficient for the entropy term.
  Note, this can be absorbed into the scale of the reward as:
  Q(s_t, a_t) = R(s_t, a_t) + alpha * H(A_t | S_t = s_t)
  Between alpha and reward_scale, one can fix one and only needs to adjust one of them,
  """
    lr: float = 3e-4  # learning rate
    gamma: float = 0.99  # discounting factor
    replay_buffer_capacity: int = 1_000_000
    tau: float = 0.005  # EMA rate for target network
    batch_size: int = 256

    lambda_v: float = 1.0
    lambda_q: float = 1.0
    lambda_pi: float = 1.0

    max_norm: float = 0.5

    ########################################
    # logging-related
    ########################################
    log_every_n_steps: int = 5_000
    project_name: str = ""  # derived from env_name in __post_init__ if empty
    run_name: str = "dbg"
    disable_wandb: bool = True

    ########################################
    # checkpoint-related options
    ########################################
    workspace_dir: str = "./workspace_rl-sac"
    ckpt_dir: str = "ckpt_dir"
    save_every_n_steps: int = 5_000
    resume: bool = True

    def get_ckpt_dir(self):
        return Path(self.workspace_dir) / self.ckpt_dir / self.run_name

    def __post_init__(self):
        if self.env_name not in SUPPORTED_ENVS:
            raise ValueError(
                f"Unsupported env '{self.env_name}'. Choose from: {SUPPORTED_ENVS}"
            )
        workspace = Path(self.workspace_dir)
        if not self.workspace_dir or not self.workspace_dir.strip():
            raise ValueError("workspace_dir must not be empty")
        if workspace.exists() and not workspace.is_dir():
            raise ValueError(
                f"workspace_dir '{self.workspace_dir}' exists but is not a directory"
            )
        workspace.mkdir(parents=True, exist_ok=True)
        if not self.project_name:
            self.project_name = f"sac-{self.env_name.lower().replace('-', '_')}"
        _ckpt_dir = self.get_ckpt_dir()
        if _ckpt_dir.exists():
            logger.warning(
                f"Checkpoint directory already exists @ {_ckpt_dir} -- "
                "Make sure you actually want to use the same checkpoint directory!!!"
            )
        else:
            _ckpt_dir.mkdir(parents=True, exist_ok=False)
            logger.info(f"workspace created @ {_ckpt_dir}")


class MetricLogger:
    """Unified metric sink for training.

    Wraps two backends behind one ``log`` call:

    * ``disable_wandb=True`` (default): append each record as one JSON object per
      line (JSONL) to ``<ckpt_dir>/metrics.jsonl``. JSONL is append-only and
      crash-safe (a truncated final line never corrupts earlier records), keeps
      numeric types intact, tolerates a changing set of metrics between runs, and
      is trivially loaded with ``pandas.read_json(path, lines=True)`` or ``jq``.
    * ``disable_wandb=False``: forward to Weights & Biases via ``wandb.log``.

    The interface mirrors the subset of the wandb API this script uses
    (``log`` / ``run_id`` / ``finish``) so call sites stay backend-agnostic.
    """

    def __init__(self, config: "Config", *, run_id: str | None = None) -> None:
        self.disabled = config.disable_wandb
        self._file = None
        if self.disabled:
            log_path = config.get_ckpt_dir() / "metrics.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered append so records are flushed as soon as they're written.
            self._file = open(log_path, "a", buffering=1, encoding="utf-8")
            logger.info(f"Logging metrics to {log_path}")
        else:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.__dict__,
                id=run_id,
                resume="must" if run_id else ("allow" if config.resume else None),
            )

    def log(self, data: dict, step: int | None = None) -> None:
        """Record one metric dict, optionally tagged with a training step."""
        if self.disabled:
            record = {"step": step, **data} if step is not None else dict(data)
            assert self._file is not None
            self._file.write(json.dumps(record) + "\n")
        else:
            wandb.log(data, step=step)

    @property
    def run_id(self) -> str | None:
        """The wandb run id (for checkpointing), or ``None`` in local mode."""
        if self.disabled:
            return None
        return wandb.run.id if wandb.run is not None else None

    def finish(self) -> None:
        """Close the local file or finish the wandb run."""
        if self.disabled:
            if self._file is not None:
                self._file.close()
                self._file = None
        else:
            wandb.finish()


def train(config: Config):
    # Checkpoint paths
    ckpt_path = config.get_ckpt_dir() / "latest.pt"
    buffer_path = config.get_ckpt_dir() / "buffer.npz"

    run_id = None
    checkpoint = None
    if config.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        run_id = checkpoint.get("run_id", None)

    # Seed all RNGs before anything stochastic happens (network weight init,
    # env resets, action sampling). On resume, offset by the resumed step so we
    # continue the stochastic stream instead of replaying it from the start.
    resumed_step = checkpoint.get("global_step", 0) if checkpoint is not None else 0
    seed = config.seed + resumed_step
    seed_everything(seed, deterministic_cudnn=config.deterministic_cudnn)

    metric_logger = MetricLogger(config, run_id=run_id)

    # Create vectorized environment
    env: gym.vector.VectorEnv = create_vectorized_env(
        env_name=config.env_name,
        max_episode_steps=config.max_episode_steps,
        num_envs=config.num_envs,
        reward_scale=config.reward_scale,
        record_every_n_episodes=config.record_every_n_episodes,
        record_video_folder="./recordings",
    )

    # environment for evaluation
    eval_env: gym.vector.VectorEnv = create_vectorized_env(
        env_name=config.env_name,
        max_episode_steps=config.max_episode_steps,
        num_envs=1,
        reward_scale=config.reward_scale,
    )

    assert isinstance(env.single_observation_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Box)

    policy = PolicyNetwork(
        obs_dim=env.single_observation_space.shape[0],
        action_dim=env.single_action_space.shape[0],
        action_low=torch.as_tensor(env.single_action_space.low, dtype=torch.float32),
        action_high=torch.as_tensor(env.single_action_space.high, dtype=torch.float32),
    ).to(config.device)
    policy.train()

    qnet1 = ActionValueNetwork(
        obs_dim=env.single_observation_space.shape[0],
        action_dim=env.single_action_space.shape[0],
    ).to(config.device)
    qnet1.train()
    qnet2 = ActionValueNetwork(
        obs_dim=env.single_observation_space.shape[0],
        action_dim=env.single_action_space.shape[0],
    ).to(config.device)
    qnet2.train()

    vnet = ValueNetwork(obs_dim=env.single_observation_space.shape[0]).to(config.device)
    vnet.train()

    vnet_target = copy.deepcopy(vnet).to(config.device)
    vnet_target.eval()

    # create the optimizer
    optim_policy = Adam(policy.parameters(), lr=config.lr)
    optim_q = Adam(list(qnet1.parameters()) + list(qnet2.parameters()), lr=config.lr)
    optim_v = Adam(vnet.parameters(), lr=config.lr)

    # convenient variables
    B = config.num_envs
    T = config.max_episode_steps
    dim_obs = env.single_observation_space.shape[0]
    dim_act = env.single_action_space.shape[0]

    buffer = VectorReplayBuffer(
        config.replay_buffer_capacity, B, dim_obs, dim_act, config.device
    )

    global_step = 0
    total_timesteps_used = 0

    if config.resume and checkpoint is not None:
        policy.load_state_dict(checkpoint["policy"])
        qnet1.load_state_dict(checkpoint["qnet1"])
        qnet2.load_state_dict(checkpoint["qnet2"])
        vnet.load_state_dict(checkpoint["vnet"])
        vnet_target.load_state_dict(checkpoint["vnet_target"])
        optim_policy.load_state_dict(checkpoint["optim_policy"])
        optim_q.load_state_dict(checkpoint["optim_q"])
        optim_v.load_state_dict(checkpoint["optim_v"])
        global_step = checkpoint["global_step"]
        total_timesteps_used = checkpoint["total_timesteps_used"]
        logger.info(f"Resumed models and optimizers from step {global_step}")

        if buffer_path.exists():
            buffer.load(str(buffer_path))

    # progress bar
    tqdm_bar = tqdm(
        total=config.max_timesteps_used, initial=total_timesteps_used, desc="Training"
    )  # Get the tqdm object

    # Seed the gym env streams. ``reset`` seeds the initial-state RNG and
    # ``action_space.seed`` the (separate) random-exploration sampler used before
    # learning starts. AsyncVectorEnv derives a distinct per-worker stream from
    # the base seed, so the workers don't share an identical trajectory.
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    for _episode in range(config.num_episodes):
        # collect enough timesteps, break
        if total_timesteps_used >= config.max_timesteps_used:
            break

        for _i in range(T):
            if total_timesteps_used >= config.max_timesteps_used:
                break

            total_timesteps_used += config.num_envs
            tqdm_bar.update(config.num_envs)

            if total_timesteps_used < config.learning_starts_at_n_timesteps:
                # IMPORTANT: To avoid overfitting to the initialization of the policy,
                # sample actions directly from the environment to ensure sufficent
                # coverage of the action space.
                _action = env.action_space.sample()
            else:
                with torch.no_grad():
                    sample = policy(
                        torch.as_tensor(obs, dtype=torch.float32, device=config.device)
                    )
                _action = sample.action.cpu().numpy()

            (next_obs, reward, terminated, truncated, info) = env.step(_action)

            # Extract the true final observations for the replay buffer
            real_next_obs = next_obs.copy()
            if "final_observation" in info:
                for env_idx in range(config.num_envs):
                    if info["_final_observation"][env_idx]:
                        real_next_obs[env_idx] = info["final_observation"][env_idx]

            # add data to the buffer
            buffer.add(
                obs, _action, reward[:, None], real_next_obs, terminated[:, None]
            )

            obs = next_obs

            # Only start learning when enough timesteps are collected,
            # otherwise, we will be always sampling from a small set of samples
            # which will cause overfitting.
            if total_timesteps_used < config.learning_starts_at_n_timesteps:
                continue

            num_batches = max(1, buffer.size // config.batch_size)
            for _ in range(min(num_batches, config.num_envs)):
                s_batch, a_batch, r_batch, snext_batch, done_batch = buffer.sample(
                    config.batch_size
                )

                # Jv term: Matching V(s_t) to the expectation E_{A_t ~ policy( |s_t)} [Q(s_t, A_t) + alpha H(A_t | s_t)]
                # s_t is sampled from the replay buffer, A_t is sampled from the optimizing policy given s_t.
                # (Why sample A_t from the optimizing policy? Because V is defined as the value function
                # under the optimizing policy)
                # However, we freeze both policy network and Q network to construct Jv -- only the parameters
                # of V network are being optimized.
                sample = policy(s_batch)
                _q1 = qnet1(s_batch, sample.action)
                _q2 = qnet2(s_batch, sample.action)
                _q = torch.min(_q1, _q2)
                Jv = 0.5 * torch.mean(
                    (
                        vnet(s_batch)
                        - (_q.detach() + config.alpha * sample.entropy.detach())
                    )
                    ** 2
                )

                # Jq term: Matching Q(s_t, a_t) to the one-step bootstrapped target
                # r_t + E_{S_{t+1} | s_t, a_t} [Vtarget(S_{t+1})]
                # Note, the expectation is taken over the transition probability P(S_{t+1} | (s_t, a_t))
                # We can sample (one) S_{t+1} from P given (s_t, a_t) to approximate the expectation.
                # Here, we use s_{t+1} which is already sampled along with s_t, a_t, and r_t and stored in
                # the replay buffer.
                with torch.no_grad():
                    # bootstrap q target
                    q_target = r_batch + config.gamma * (1 - done_batch) * vnet_target(
                        snext_batch
                    )
                Jq1 = 0.5 * torch.mean((qnet1(s_batch, a_batch) - q_target) ** 2)
                Jq2 = 0.5 * torch.mean((qnet2(s_batch, a_batch) - q_target) ** 2)

                # Jpi term: Matching policy(A_t | s_t) to a target policy.
                # The target policy the normalized expontential of action value, that is
                # P(A_t | s_t) = exp(Q(s_t, A_t)) / Z(s_t) where Z is a normalizing factor.
                Jpi = torch.mean(sample.log_prob * config.alpha - _q)

                loss_v = config.lambda_v * Jv
                loss_q = config.lambda_q * (Jq1 + Jq2)
                loss_pi = config.lambda_pi * Jpi

                optim_policy.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=config.max_norm
                )
                optim_policy.step()

                optim_v.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(
                    vnet.parameters(), max_norm=config.max_norm
                )
                optim_v.step()

                optim_q.zero_grad()
                loss_q.backward()
                for net in [qnet1, qnet2]:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), max_norm=config.max_norm
                    )
                optim_q.step()

                with torch.no_grad():
                    for param, target_param in zip(
                        vnet.parameters(), vnet_target.parameters(), strict=False
                    ):
                        target_param.data.copy_(
                            config.tau * param.data
                            + (1.0 - config.tau) * target_param.data
                        )

                global_step += 1

                if global_step % config.log_every_n_steps == 0:
                    # Update tqdm postfix with the losses and log to wandb

                    # compute reward using a frozen policy every N steps
                    # currently the reward is computed using a changing policy
                    policy.eval()
                    eval_result = evaluate_policy(
                        eval_env=eval_env,
                        policy=policy,
                        T=T,
                        device=config.device,
                        seed=config.seed + 10_000,
                    )
                    policy.train()

                    log_item = {
                        "global_step": global_step,
                        "loss_v": Jv.detach().cpu().item(),
                        "loss_q1": Jq1.detach().cpu().item(),
                        "loss_q2": Jq2.detach().cpu().item(),
                        "loss_pi": Jpi.detach().cpu().item(),
                        "timesteps_used": total_timesteps_used,
                        "cumulative_reward": eval_result.average_cumulative_reward
                        / config.reward_scale,
                        "entropy": eval_result.entropy,
                    }
                    tqdm_bar.set_postfix(
                        {
                            "global_step": f"{log_item['global_step']}",
                            "samples": f"{log_item['timesteps_used']}",
                            "cum_reward": f"{log_item['cumulative_reward']}",
                            "entropy": f"{log_item['entropy']}",
                        }
                    )
                    metric_logger.log(log_item, step=global_step)

                if global_step % config.save_every_n_steps == 0:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "qnet1": qnet1.state_dict(),
                            "qnet2": qnet2.state_dict(),
                            "vnet": vnet.state_dict(),
                            "vnet_target": vnet_target.state_dict(),
                            "optim_policy": optim_policy.state_dict(),
                            "optim_q": optim_q.state_dict(),
                            "optim_v": optim_v.state_dict(),
                            "global_step": global_step,
                            "total_timesteps_used": total_timesteps_used,
                            "run_id": metric_logger.run_id,
                        },
                        ckpt_path,
                    )
                    logger.info(f"Checkpoint saved to {ckpt_path}")
                    buffer.save(str(buffer_path))

            # end-of-batch-for-loop
    ########################################
    # clean up
    ########################################
    tqdm_bar.close()
    env.close()
    eval_env.close()
    metric_logger.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on a MuJoCo environment")
    parser.add_argument(
        "--env-name",
        type=str,
        default="HalfCheetah-v5",
        choices=SUPPORTED_ENVS,
        help="MuJoCo environment to train on",
    )
    parser.add_argument(
        "--run-name", type=str, default="autoresearch-dbg", help="Name for this run"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for all RNGs")
    parser.add_argument(
        "--resume", action="store_true", default=True, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="Start fresh"
    )
    parser.add_argument("--disable-wandb", action="store_true", default=True)
    parser.add_argument(
        "--max-timesteps-used", type=int, default=1_000_000, help="Training budget"
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="./workspace_rl-sac",
        help="Root directory for checkpoints and logs (created if absent)",
    )
    parser.add_argument(
        "--learning-starts-at-n-timesteps",
        type=int,
        default=10_000,
        help="Timesteps of random exploration before learning begins",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1_000,
        help="Gradient steps between metric log entries",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train(
        Config(
            env_name=args.env_name,
            run_name=args.run_name,
            seed=args.seed,
            resume=args.resume,
            disable_wandb=args.disable_wandb,
            max_timesteps_used=args.max_timesteps_used,
            workspace_dir=args.workspace_dir,
            learning_starts_at_n_timesteps=args.learning_starts_at_n_timesteps,
            log_every_n_steps=args.log_every_n_steps,
            save_every_n_steps=50_000,
        )
    )


if __name__ == "__main__":
    main()

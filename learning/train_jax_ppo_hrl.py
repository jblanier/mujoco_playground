# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO-HRL agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo_hrl import networks as ppo_hrl_networks
from brax.training.agents.ppo_hrl import train as ppo_hrl
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX

try:
  import wandb
except ImportError:
  wandb = None


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project",
    "mjxrl",
    "Weights & Biases project name",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)

# HRL-specific flags
_HINT_NVEC = flags.DEFINE_string(
    "hint_nvec", "5,5", "Hint space dimensions (comma-separated, categorical only)"
)
_HIGH_ENTROPY_COST = flags.DEFINE_float(
    "high_entropy_cost", 1e-4, "Entropy cost for high-level policy"
)
_MI_LOW_COEF = flags.DEFINE_float(
    "mi_low_coef", 0.1, "MI regularization for low-level policy"
)
_MI_HIGH_COEF = flags.DEFINE_float(
    "mi_high_coef", 0.1, "MI regularization for high-level policy"
)
_DEBUG_CONSTANT_HINTS = flags.DEFINE_boolean(
    "debug_constant_hints",
    False,
    "Use constant zero hints (bypass high policy). Useful for debugging - "
    "makes low policy equivalent to standard PPO with constant input appended.",
)
_DEBUG_PASSTHROUGH_LOW_AGENT = flags.DEFINE_boolean(
    "debug_passthrough_low_agent",
    False,
    "Low policy passes hints through as actions. Auto-enables continuous hints "
    "with hint_dim=action_size. Useful for testing high policy in isolation.",
)
_CONTINUOUS_HINTS = flags.DEFINE_boolean(
    "continuous_hints",
    False,
    "Use continuous hints (NormalTanhDistribution) instead of categorical "
    "(MultiCategoricalDistribution).",
)
_CONTINUOUS_HINT_DIM = flags.DEFINE_integer(
    "continuous_hint_dim",
    None,
    "Dimension of continuous hint space. Required if continuous_hints=True "
    "and not using debug_passthrough_low_agent.",
)
_CONTINUOUS_HINT_INIT_SCALE = flags.DEFINE_float(
    "continuous_hint_init_scale",
    1.0,
    "Initial std scale for continuous hint distribution.",
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  # PPO-HRL does not support vision mode, so we only use the state-based configs
  if env_name in mujoco_playground.manipulation._envs:
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env_cfg_overrides = {}
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides = json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value)
  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  # Parse HRL-specific params
  hint_nvec = tuple(int(x) for x in _HINT_NVEC.value.split(","))

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")
  print(f"HRL Parameters: hint_nvec={hint_nvec}, high_entropy_cost="
        f"{_HIGH_ENTROPY_COST.value}, mi_low_coef={_MI_LOW_COEF.value}, "
        f"mi_high_coef={_MI_HIGH_COEF.value}")
  print(f"Continuous hints: {_CONTINUOUS_HINTS.value}, dim={_CONTINUOUS_HINT_DIM.value}, "
        f"init_scale={_CONTINUOUS_HINT_INIT_SCALE.value}")
  if _DEBUG_CONSTANT_HINTS.value:
    print("\n*** DEBUG_CONSTANT_HINTS MODE ENABLED ***")
    print("High policy is bypassed - using constant zero hints.")
    print("Low policy should behave like standard PPO with constant input appended.\n")
  if _DEBUG_PASSTHROUGH_LOW_AGENT.value:
    print("\n*** DEBUG_PASSTHROUGH_LOW_AGENT MODE ENABLED ***")
    print("Low policy passes hints through as actions.")
    print("High policy learns in isolation - should match standard PPO performance.\n")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-ppo_hrl-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. "
          "Install via: pip install wandb"
      )
    wandb.init(project=_WANDB_PROJECT.value, name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({
        "env_name": _ENV_NAME.value,
        "hint_nvec": hint_nvec,
        "high_entropy_cost": _HIGH_ENTROPY_COST.value,
        "mi_low_coef": _MI_LOW_COEF.value,
        "mi_high_coef": _MI_HIGH_COEF.value,
        "debug_constant_hints": _DEBUG_CONSTANT_HINTS.value,
        "debug_passthrough_low_agent": _DEBUG_PASSTHROUGH_LOW_AGENT.value,
        "continuous_hints": _CONTINUOUS_HINTS.value,
        "continuous_hint_dim": _CONTINUOUS_HINT_DIM.value,
        "continuous_hint_init_scale": _CONTINUOUS_HINT_INIT_SCALE.value,
    })

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  # Use PPO-HRL network factory
  network_fn = ppo_hrl_networks.make_ppo_hrl_networks
  network_kwargs = {
      "hint_nvec": hint_nvec,
      "continuous_hints": _CONTINUOUS_HINTS.value,
      "continuous_hint_dim": _CONTINUOUS_HINT_DIM.value,
      "continuous_hint_init_scale": _CONTINUOUS_HINT_INIT_SCALE.value,
      "debug_passthrough_low_agent": _DEBUG_PASSTHROUGH_LOW_AGENT.value,
  }
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn,
        **network_kwargs,
        **ppo_params.network_factory
    )
  else:
    network_factory = functools.partial(network_fn, **network_kwargs)

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo_hrl.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      # HRL-specific params
      hint_nvec=hint_nvec,
      high_entropy_cost=_HIGH_ENTROPY_COST.value,
      mi_low_coef=_MI_LOW_COEF.value,
      mi_high_coef=_MI_HIGH_COEF.value,
      # Debug modes
      debug_constant_hints=_DEBUG_CONSTANT_HINTS.value,
      debug_passthrough_low_agent=_DEBUG_PASSTHROUGH_LOW_AGENT.value,
      # Continuous hints
      continuous_hints=_CONTINUOUS_HINTS.value,
      continuous_hint_dim=_CONTINUOUS_HINT_DIM.value,
      continuous_hint_init_scale=_CONTINUOUS_HINT_INIT_SCALE.value,
  )

  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()

    # Print all metrics in YAML-style format
    print(f"\n{'='*60}")
    print(f"step: {num_steps}")
    print(f"{'='*60}")

    # Group metrics by prefix for cleaner outputno
    grouped = {}
    for key, value in sorted(metrics.items()):
      parts = key.split('/')
      if len(parts) > 1:
        group = parts[0]
        subkey = '/'.join(parts[1:])
      else:
        group = 'misc'
        subkey = key
      if group not in grouped:
        grouped[group] = {}
      grouped[group][subkey] = value

    for group, group_metrics in sorted(grouped.items()):
      print(f"{group}:")
      for key, value in sorted(group_metrics.items()):
        if isinstance(value, float):
          print(f"  {key}: {value:.6f}")
        else:
          print(f"  {key}: {value}")

  # Load evaluation environment.
  eval_env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  num_envs = 1

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    # Note: rscope integration for HRL may need custom handling
    print("Warning: rscope integration may have limited support for PPO-HRL")

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Create inference function.
  # PPO-HRL inference requires prev_hints argument
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Determine hint configuration for rollouts
  # (mirrors logic in train.py and networks.py)
  rollout_continuous_hints = _CONTINUOUS_HINTS.value or _DEBUG_PASSTHROUGH_LOW_AGENT.value
  if rollout_continuous_hints:
    if _DEBUG_PASSTHROUGH_LOW_AGENT.value:
      rollout_hint_dim = eval_env.action_size
    else:
      rollout_hint_dim = _CONTINUOUS_HINT_DIM.value
    rollout_hint_dtype = jp.float32
  else:
    rollout_hint_dim = len(hint_nvec)
    rollout_hint_dtype = jp.int32

  # Run evaluation rollouts.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    # Initialize prev_hints for HRL (dtype depends on continuous vs categorical)
    prev_hints = jp.zeros((rollout_hint_dim,), dtype=rollout_hint_dtype)

    def step(carry, _):
      state, prev_hints, rng = carry
      rng, act_key = jax.random.split(rng)

      # HRL inference takes (obs, prev_hints, key) and returns (action, extras)
      if isinstance(state.obs, dict):
        obs = state.obs['state']
      else:
        obs = state.obs
      act, extras = jit_inference_fn(obs, prev_hints, act_key)

      state = eval_env.step(state, act)

      # Update prev_hints from the policy output, reset on done
      new_hints = extras['hint']
      new_hints = jp.where(state.done, jp.zeros_like(new_hints), new_hints)

      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      return (state, new_hints, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, prev_hints, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as 'rollout{i}.mp4'.")


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()

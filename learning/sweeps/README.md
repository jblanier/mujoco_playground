# PPO-HRL Sweep

Hyperparameter sweep for PPO-HRL and baseline PPO comparison.

## Environment Setup

Reproduce the conda environment on other machines:

```bash
conda env create -f environment.yaml
conda activate brax
cd <brax repo>
pip install -e .
cd <mujoco playground repo>
pip install -e ".[all]"                                                                                                      
```

## Available Environments

DM Control Suite environments available via `--env_name`:

| Environment | Description |
|-------------|-------------|
| `CartpoleBalance` | Balance a pole on a cart |
| `CartpoleBalanceSparse` | CartpoleBalance with sparse reward |
| `CartpoleSwingup` | Swing up and balance a pole |
| `CartpoleSwingupSparse` | CartpoleSwingup with sparse reward |
| `AcrobotSwingup` | Swing up a two-link pendulum |
| `AcrobotSwingupSparse` | AcrobotSwingup with sparse reward |
| `BallInCup` | Catch a ball in a cup |
| `CheetahRun` | Make a cheetah run |
| `FingerSpin` | Spin an object with a finger |
| `FingerTurnEasy` | Turn an object to target (easy) |
| `FingerTurnHard` | Turn an object to target (hard) |
| `FishSwim` | Control a fish to swim |
| `HopperHop` | Make a hopper hop |
| `HopperStand` | Make a hopper stand |
| `HumanoidStand` | Make a humanoid stand |
| `HumanoidWalk` | Make a humanoid walk |
| `HumanoidRun` | Make a humanoid run |
| `PendulumSwingup` | Swing up a pendulum |
| `PointMass` | Move a point mass to target |
| `ReacherEasy` | Reach a target (easy) |
| `ReacherHard` | Reach a target (hard) |
| `SwimmerSwimmer6` | Control a 6-link swimmer |
| `WalkerStand` | Make a walker stand |
| `WalkerWalk` | Make a walker walk |
| `WalkerRun` | Make a walker run |

## PPO Baseline

Run standard PPO with default hyperparameters for comparison:

```bash
cd <playground repo location>/mujoco_playground/learning

# Local
python train_jax_ppo.py --env_name=CartpoleBalance --use_wandb

# SLURM (modify partition/node as needed)
sbatch --partition=<partition> --nodelist=<node> --gres=gpu:1 --wrap="python train_jax_ppo.py --env_name=CartpoleBalance --use_wandb"
```

This uses the tuned defaults from `dm_control_suite_params.py` (60M steps, lr=1e-3, etc.).

## PPO-HRL (Full Hierarchical)

Run PPO-HRL with both high and low policies learning together:

```bash
python train_jax_ppo_hrl.py --env_name=CartpoleBalance \
    --hint_nvec=5,5 \
    --high_entropy_cost=1e-4 \
    --mi_low_coef=0.1 \
    --mi_high_coef=0.1 \
    --use_wandb
```

### HRL-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hint_nvec` | `5,5` | Categorical hint space dimensions. Each value is the number of categories for that hint dimension. `5,5` = 2 hint dimensions with 5 categories each (25 possible hints). |
| `--high_entropy_cost` | `1e-4` | Entropy bonus for high policy. Encourages exploration in hint space. |
| `--mi_low_coef` | `0.1` | Mutual information regularization for low policy. Encourages low policy to use hints. |
| `--mi_high_coef` | `0.1` | Mutual information regularization for high policy. Encourages high policy to produce informative hints. |

## PPO-HRL Debug Modes

PPO-HRL has two debug modes for testing components in isolation:

### `--debug_passthrough_low_agent` (Test High Policy)

The low policy becomes a passthrough: it outputs the hint directly as the action. This lets the high policy control actions directly, equivalent to standard PPO but going through the HRL code path.

**Use case:** Verify the high policy can learn. Performance should match standard PPO.

```bash
python train_jax_ppo_hrl.py --env_name=CartpoleBalance --debug_passthrough_low_agent --use_wandb
```

### `--debug_constant_hints` (Test Low Policy)

The high policy is bypassed and hints are fixed to zero. The low policy receives constant zero hints appended to observations, making it equivalent to standard PPO with extra constant input features.

**Use case:** Verify the low policy can learn independently. Should also set `--mi_low_coef=0` since MI loss doesn't make sense with constant hints.

```bash
python train_jax_ppo_hrl.py --env_name=CartpoleBalance --debug_constant_hints --mi_low_coef=0 --use_wandb
```

### Expected Results

Both debug modes should achieve similar performance to standard PPO on the same environment, since they reduce the hierarchical structure to essentially flat RL.

## PPO-HRL Sweep

### 1. Create the sweep

```bash
cd /home/jb/git/mujoco_playground/learning/sweeps
wandb sweep jan_25/ppo_hrl_cartpole_balance.yaml
```

This outputs a sweep ID like `username/mjxrl-sweep/abc123`.

### 2. Run sweep agents

**Local:**
```bash
wandb agent <sweep_id>
```

**SLURM:**
```bash
sbatch --partition=<partition> --nodelist=<node> sweep_agent.sbatch <sweep_id>
```

### 3. Monitor

View results at https://wandb.ai/<username>/mjxrl-sweep

## Sweep Configuration

### Search Parameters

The sweep optimizes `eval/episode_reward` using Bayesian search over:

**PPO Parameters:**

| Parameter | Search Space |
|-----------|-------------|
| `learning_rate` | log-uniform [1e-4, 1e-2] |
| `entropy_cost` | log-uniform [1e-4, 1e-1] |
| `unroll_length` | [10, 20, 30, 50] |
| `num_minibatches` | [16, 32, 64] |
| `num_updates_per_batch` | [4, 8, 16, 32] |
| `discounting` | [0.99, 0.995, 0.999] |
| `reward_scaling` | log-uniform [0.1, 100] |
| `clipping_epsilon` | [0.1, 0.2, 0.3] |

**HRL Parameters (Discrete Hints):**

| Parameter | Search Space |
|-----------|-------------|
| `hint_nvec` | ["3,3", "5,5", "4,4,4", "8,8"] |
| `high_entropy_cost` | log-uniform [1e-5, 1e-2] |
| `mi_low_coef` | log-uniform [0.01, 1.0] |
| `mi_high_coef` | log-uniform [0.01, 1.0] |

### Fixed Parameters

- `env_name`: CartpoleBalance
- `num_timesteps`: 20,000,000
- `wandb_project`: mjxrl-sweep (configurable in YAML)

### SLURM Resources

Default resource requests in `sweep_agent.sbatch`:
- 1 GPU
- 8 CPUs
- 32GB memory
- 72 hour time limit

Modify the `#SBATCH` directives as needed.

## Customization

**Prefer creating new sweep files** rather than modifying existing ones. This preserves a record of past experiments. Name files with date suffix (MMDD format):

```
jan_25/ppo_hrl_cartpole_balance.yaml    # January 25
sweep_walker_run_0202.yaml # February 2
```

### Create a new sweep config

Copy an existing config and modify as needed:

```bash
cp jan_25/ppo_hrl_cartpole_balance.yaml sweep_my_experiment_0126.yaml
```

Then edit the new file to change:

- `wandb_project` - Project name for grouping runs
- `env_name` - Environment to train on
- `parameters` - Search space for hyperparameters

### Activate conda in SLURM

Uncomment and modify the activation line in `sweep_agent.sbatch`:
```bash
source ~/miniconda3/bin/activate brax
```

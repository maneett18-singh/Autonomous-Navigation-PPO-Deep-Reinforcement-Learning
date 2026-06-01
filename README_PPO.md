# PPO Continuous Control (Multi-seed)

Runs PPO training across multiple seeds for the race car environment (`ppo_continuos_seed_10.py`).

## Installation

```bash
pip install gymnasium torch numpy matplotlib tensorboard wandb
```

## Usage

**Run default experiment (10 seeds, parcour):**
```bash
python ppo_continuos_seed_10.py
```

**Run with custom settings:**
```bash
python ppo_continuos_seed_10.py --env-type curvy --num-seeds 3 --track
```

## Reference
Code adapted from: https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py

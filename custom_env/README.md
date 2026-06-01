# Archery RL Environment

A custom Reinforcement Learning environment where an agent learns to shoot arrows at a target, accounting for gravity, distance, and variable wind conditions. Built with **Gymnasium**, **Pygame**, and **Stable Baselines 3**.

## 🎯 Project Overview
The goal is to hit a target (red circle) with an arrow. The agent controls the **angle** and **power** of the shot. The environment simulates 2D projectile physics including:
- **Gravity**: Pulls the arrow down over time.
- **Wind**: Random horizontal force that affects trajectory.
- **One-Shot Dynamics**: Agent takes one action (aim & power), then the physics engine simulates the flight.

## 🛠️ Installation

1. **Prerequisites**: Python 3.8+
2. **Install Dependencies**:
   ```bash
   cd arrow
   pip install -r requirements.txt
   ```
   *Required packages: `gymnasium`, `stable-baselines3`, `pygame`, `numpy`, `imageio`, `moviepy`, `shimmy`*

## 🚀 Usage

### 1. Manual Play (Test Yourself)
Try to hit the target yourself using keyboard controls!
```bash
python manual.py
```
**Controls**:
- `W` / `S`: Adjust Aim Angle
- `SPACE` (Hold): Charge Power
- `SPACE` (Release): Fire

### 2. Train the AI Agent
Train a PPO (Proximal Policy Optimization) agent to solve the environment.
```bash
python train.py
```
- Trains for 50,000 steps by default.
- Saves the best model to `models/PPO/`.
- Saves training videos to `logs/videos/` periodically.
- Plots a learning curve at the end.

### 3. Evaluate the AI
Watch the trained agent play.
```bash
python evaluate.py
```

## 🧠 Environment Details

### Observation Space (5 values)
The agent sees a vector containing:
1. **Target X**: Horizontal position of target.
2. **Target Y**: Vertical position of target.
3. **Distance**: Straight-line distance to target.
4. **Target Angle**: Ideal angle to aim directly at target.
5. **Wind**: Wind speed (Positive = Tailwind, Negative = Headwind).

### Action Space (2 values)
Continuous values between -1 and 1:
1. **Angle**: Mapped to 0° - 85°.
2. **Power**: Mapped to 0 - 35 m/s velocity.

### Reward Function
- **Dense Reward**: Negative of the *minimum distance* the arrow achieved relative to the target center during flight (encourages getting close).
- **Target Hit**: +100 bonus points if the arrow collides with the target.

## 📂 File Structure
- `env.py`: The custom Gym environment logic.
- `utils.py`: Physics constants and helper functions.
- `train.py`: Training script using SB3 PPO.
- `evaluate.py`: Script to load and visualize trained models.
- `manual.py`: Playable interactive version.

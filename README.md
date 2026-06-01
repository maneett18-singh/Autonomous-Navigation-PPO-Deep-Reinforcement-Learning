# 🏎️ Autonomous Racing Vehicle Navigation with Deep Reinforcement Learning (PPO)

A continuous-control reinforcement learning project that trains an autonomous racing vehicle to navigate a complex track using **Proximal Policy Optimization (PPO)**. The agent learns to drive through sequential checkpoints while respecting realistic vehicle dynamics, velocity-dependent steering constraints, and stochastic control noise.

---

## 🎯 Project Overview

The goal is to develop an autonomous agent capable of efficiently navigating a challenging race track composed of **34 sequential gates**. Unlike traditional path-planning approaches, the agent learns a driving policy directly through interaction with the environment.

### Key Challenges

* Continuous action space
* Non-holonomic vehicle dynamics
* Sharp turns and 180° hairpin corners
* Velocity-dependent steering limitations
* Stochastic acceleration noise
* Long-horizon decision making

The resulting policy learns when to accelerate, brake, and rotate to successfully complete the track while minimizing time and avoiding failure conditions.

---

## 📺 Demonstration

<video src="docs/videos/parcour_demo.mp4" width="700" controls>
Your browser does not support the video tag.
</video>

> If the video preview is not displayed, the raw file can be found in `docs/videos/parcour_demo.mp4`.

---

# 🛠️ Installation

## Requirements

* Python 3.8+
* Gymnasium
* PyTorch
* NumPy
* Matplotlib
* TensorBoard
* Weights & Biases (optional)

## Install Dependencies

```bash
pip install gymnasium torch numpy matplotlib tensorboard wandb
```

---

# 🚀 Training

## Default Training Run

Execute training across 10 independent random seeds:

```bash
python ppo_continuos_seed_10.py
```

## Custom Configuration

Run with a different track configuration and seed count:

```bash
python ppo_continuos_seed_10.py \
    --env-type curvy \
    --num-seeds 3 \
    --track
```

---

# 🧠 Reinforcement Learning Algorithm

The project uses **Proximal Policy Optimization (PPO)**, an Actor-Critic policy-gradient algorithm designed for stable and sample-efficient learning in continuous action spaces.

## Actor-Critic Architecture

### Policy Network (Actor)

Learns a policy:
$$
\pi_{\theta}(a|s)
$$

that maps observations to continuous acceleration commands.

### Value Network (Critic)

Estimates:
$$
V_{\phi}(s)
$$

which serves as a baseline during policy optimization.

Both networks are implemented as multilayer perceptrons (MLPs) with **Tanh** activations.

---

## PPO Clipped Objective

To prevent unstable policy updates, PPO optimizes the clipped surrogate objective:
$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

$$
L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[
\min\left(
r_t(\theta)\hat A_t,
	ext{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

where:

* $\hat A_t$ is the Generalized Advantage Estimate (GAE)
* $\epsilon$ defines the clipping threshold

The overall optimization objective combines:

* Policy loss
* Value-function loss
* Entropy regularization

---

# 🏎️ Environment Design

## Vehicle Dynamics

The agent controls a non-holonomic vehicle moving through a sequence of gates on a custom racing circuit.

### Stochastic Control Noise

To simulate real-world uncertainty, Gaussian noise is added to acceleration commands:
$$
\sigma_{trans}=\frac{a_{limit}}{5}
$$

$$
\sigma_{rot}=\frac{\alpha_{limit}}{5}
$$

This encourages the policy to learn robust driving behavior.

---

## Velocity-Dependent Steering Constraints

The maximum rotational velocity decreases as forward velocity increases:
$$
v_{rot,limit}=V_{rot,max}\left(1-(1-k)\frac{|v_{trans}|}{V_{trans,max}}\right)
$$

with:
$$
k = 0.3
$$

### Consequence

The agent cannot make aggressive turns at high speed and must learn to brake before entering sharp corners.

---

# 📊 Observation Space

Rather than providing global coordinates, observations are expressed relative to the next target gate.

## Observation Vector
$$
s'=[dist,\sin(\phi),\cos(\phi),v,\omega]^T
$$

where:

| Variable | Description            |
| -------- | ---------------------- |
| $dist$   | Distance to next gate  |
| $\phi$   | Heading error          |
| $v$      | Translational velocity |
| $\omega$ | Rotational velocity    |

Using $\sin(\phi)$ and $\cos(\phi)$ removes angular discontinuities at $\pm\pi$ and improves learning stability.

---

# 🎮 Action Space

The policy outputs two continuous control signals:

## Translational Acceleration
$$
a_{trans}\in[-0.5,0.5]\ \mathrm{m/s^2}
$$

## Rotational Acceleration
$$
a_{rot}\in[-0.2,0.2]\ \mathrm{rad/s^2}
$$

Vehicle states are propagated using Euler integration with:
$$
\Delta T = 0.2\,s
$$

---

# 🎯 Reward Function

A dense reward function is used to accelerate learning:
$$
r_t=r_{gate}+w_{dist}(d_{t-1}-d_t)+w_{align}(v\cos\phi)-w_{time}\,\mathcal{P}
$$

### Reward Components

✅ Passing gates

✅ Reducing distance to target

✅ Aligning velocity toward target

❌ Excessive angular motion

❌ Remaining idle

❌ Wasting time

This reward structure encourages smooth and efficient navigation behavior.

---

# ⛔ Episode Termination

An episode terminates if:

* The vehicle remains stagnant for 200 steps
* The vehicle moves more than 15 meters away from the track
* The maximum episode length is reached

---

# 📈 Results

Performance aggregated over **10 independent training seeds**.

| Metric                                                  | Mean ± SD      |
| ------------------------------------------------------- | -------------- |
| Calls to `calc_next_state` until first successful solve | 6969.8 ± 483.1 |
| Shortest successful episode                             | 386 steps      |
| Calls to `calc_next_state` until best solution          | 83375.1        |

These results demonstrate that PPO successfully learns a robust navigation policy capable of handling nonlinear constraints and stochastic dynamics.

---

# 🔬 Features

* PPO-based continuous control
* Goal-relative state representation
* Velocity-dependent steering limits
* Stochastic dynamics
* Dense reward shaping
* Multi-seed benchmarking
* TensorBoard integration
* Weights & Biases logging

---

# 👨‍💻 Author

**Maneet Singh**

Deep Reinforcement Learning
OTH Amberg-Weiden
Winter Semester 2025

---

**Keywords:** Reinforcement Learning, PPO, Autonomous Driving, Continuous Control, Navigation, Robotics, Deep RL, PyTorch, Gymnasium

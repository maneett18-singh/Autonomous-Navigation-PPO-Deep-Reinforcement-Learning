import gymnasium as gym
from gymnasium import spaces
import numpy as np
from race_acc import StraightRace, CurvyRace, Parcour # Import the fixed classes
from race_acc import ACC_TRANS_LIMIT, ACC_ROT_LIMIT

class RLWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    def __init__(self, env=None, 
                 env_type="straight", 
                 render_mode=None,
                 w_dist=2.0,
                w_dist_abs=0.01,
                w_align=0.1,
                w_speed=0.03,
                w_ang=0.05,
                w_time=0.02,
                w_idle=0.1,  
                     ):
        super(RLWrapper, self).__init__()
        # 1. Initialize the original environment
        # - If an env instance is provided, use it.
        # - Otherwise, create one from env_type.
        self.render_mode = render_mode
        if env is not None:
            self.env = env
        else:
            if env_type == "straight":
                self.env = StraightRace()
            elif env_type == "curvy":
                self.env = CurvyRace()
            elif env_type == "parcour":
                self.env = Parcour() 
            else:
                raise ValueError(f"Unknown env_type: {env_type}")

        # 2. Define Action Space (Internal -0.1 to 0.1 for accels)
        # self.action_space = spaces.Box(
        #     low=np.array([-ACC_TRANS_LIMIT, -ACC_ROT_LIMIT], dtype=np.float32),
        #     high=np.array([+ACC_TRANS_LIMIT, +ACC_ROT_LIMIT], dtype=np.float32),
        #     shape=(2,),
        #     dtype=np.float32,
        # )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 3. Define Obs Space (Relative Dist, Relative Angle, V_trans, V_rot)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # 4. Episode-level accumulators for logging
        self.episode_dist_reward = 0.0
        self.episode_alignment_reward = 0.0
        self.episode_raw_reward = 0.0
        self.episode_length = 0
        self.current_step = 0
        self.expected_max_steps = 200 * len(self.env.get_gates())
        self.no_progress_steps = 0
        self.max_episode_steps = self.expected_max_steps * 2  # Set a max step limit for truncation

        # 5. Reward Weights
        self.w_dist = w_dist
        self.w_dist_abs = w_dist_abs
        self.w_align = w_align
        self.w_speed = w_speed
        self.w_ang = w_ang
        self.w_time = w_time
        self.w_idle = w_idle

    def _get_obs(self):
        state = self.env.state  # [x, y, theta, v, w]
        gates = self.env.get_gates()
        idx = self.env.get_gate_idx()

        if idx < len(gates):
            gate_center = np.mean(gates[idx], axis=0)

            dx = gate_center[0] - state[0]
            dy = gate_center[1] - state[1]

            dist = np.sqrt(dx**2 + dy**2)

            angle_to_gate = np.arctan2(dy, dx) - state[2]

            # ✅ Normalize angle here
            angle_to_gate = np.arctan2(
                np.sin(angle_to_gate),
                np.cos(angle_to_gate)
            )

            return np.array([
                dist,
                np.sin(angle_to_gate),
                np.cos(angle_to_gate),
                state[3],  # v
                state[4],  # w
            ], dtype=np.float32)

        return np.zeros(5, dtype=np.float32)

    
    def close(self):
        # Close matplotlib figures created by raw_env.plot() / render()
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    
    def render(self):
        if self.render_mode != "rgb_array":
            return None

        # Render offscreen (safe on headless Linux)
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        # Draw using your existing matplotlib plot() function
        self.env.plot()

        fig = plt.gcf()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].astype(np.uint8)
        return rgb

    def step(self, action):
        self.current_step += 1
        # 1. We call the original step function
        # obs_raw: [x, y, theta, v, w] from the underlying race_acc.py
        obs_raw, reward_raw, done = self.env.step(action)
        
        # 2. Get the new relative observation
        # new_obs: [dist, sin_angle, cos_angle, v, w]
        new_obs = self._get_obs()
        dist, sin_theta, cos_theta, v, w = new_obs

        delta_dist = self.prev_dist - dist
        self.prev_dist = dist

        dist_reward = self.w_dist * delta_dist - self.w_dist_abs * dist
        alignment_reward = self.w_align * cos_theta if v > 0.1 else -self.w_align
        speed_reward = self.w_speed * v
        angular_penalty = -self.w_ang * abs(w)
        time_penalty = -self.w_time
        idle_penalty = -self.w_idle if v < 0.1 else 0.0


        total_reward = (
            reward_raw
            + dist_reward
            + alignment_reward
            # + speed_reward
            + angular_penalty
            + time_penalty
            + idle_penalty
        )
        truncated = False

        if self.current_step >= self.max_episode_steps:
            truncated = True

        if new_obs[0] > 15.0:
            truncated = True
            total_reward -= 10.0

        if delta_dist <= 0.05:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps > 200:
            truncated = True
            total_reward -= 5.0

        
        # 6. Accumulate episode-level rewards
        self.episode_dist_reward += dist_reward
        self.episode_alignment_reward += alignment_reward
        self.episode_raw_reward += reward_raw
        self.episode_length += 1
        
        # 7. Prepare Info and Return
        gates_passed = int(self.env.get_gate_idx())
        n_gates = len(self.env.get_gates())
        
        info = {
            "gates_passed": gates_passed,
            "n_gates": n_gates,
            "dist_reward": dist_reward,
            "alignment_reward": alignment_reward,
            "total_reward": total_reward,
        }
        
        # Add episode summary when episode ends
        if done or truncated:
            info["episode_dist_reward"] = self.episode_dist_reward
            info["episode_alignment_reward"] = self.episode_alignment_reward
            info["episode_raw_reward"] = self.episode_raw_reward
            info["episode_gates_passed"] = gates_passed
            info["episode_gates_total"] = n_gates
        
        # We return 5 values for Gymnasium compatibility
        return new_obs, total_reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs_raw = self.env.reset()
        obs = self._get_obs()
        self.prev_dist = obs[0]
        
        # Reset episode accumulators
        self.episode_dist_reward = 0.0
        self.episode_alignment_reward = 0.0
        self.episode_raw_reward = 0.0
        self.episode_length = 0
        
        # Reset step counters (critical for truncation logic)
        self.current_step = 0
        self.no_progress_steps = 0
        
        return obs, {}
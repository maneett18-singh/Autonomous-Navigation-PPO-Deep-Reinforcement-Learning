import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from env import ArcheryEnv

class VideoRecorderCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str):
        super().__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.video_dir = os.path.join(log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env
            # Access the original env (unwrap Monitor/DummyVecEnv)
            if hasattr(env, 'envs'):
                env = env.envs[0]
            if hasattr(env, 'env'):
                env = env.env
                
            # Trigger a save video from the LAST episode trajectory
            # Since training happens constantly, we just snapshot the last trajectory
            # However, during training, PPO might collect many steps. 
            # We want to force a Render/Save of an Evaluation Run.
             
            # Better way: Create a separate eval env instance to record clean videos
            eval_env = ArcheryEnv(render_mode=None)
            obs, _ = eval_env.reset()
            # Predict with current model
            action, _ = self.model.predict(obs)
            eval_env.step(action)
            
            save_path = os.path.join(self.video_dir, f"step_{self.n_calls}.mp4")
            eval_env.save_video(save_path)
            eval_env.close()
            
        return True

def plot_results(log_folder, title='Learning Curve'):
    """Plots reward vs timesteps."""
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    if len(x) < 2: return
    
    # Smooth the curve
    window = 50
    y_av = np.convolve(y, np.ones(window)/window, mode='valid')
    x_av = x[len(x) - len(y_av):]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.3, label='Raw Reward')
    plt.plot(x_av, y_av, color='r', linewidth=2, label='Moving Avg')
    plt.xlabel('Episodes') # Since each step is an episode
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")

def train():
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # We use render_mode=None for speed during training
    env = ArcheryEnv(render_mode=None)
    env = Monitor(env, log_dir) 

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0005,
        n_steps=512,       # Smaller buffer since episodes are length 1
        batch_size=64,
        gamma=0.9,         # Lower gamma since future rewards don't exist (1-step)
        ent_coef=0.02      # Exploration is key
    )

    print("Starting training...")
    
    # Save video every 5000 steps
    video_callback = VideoRecorderCallback(check_freq=5000, log_dir=log_dir)
    
    # 50,000 shots is plenty to learn this
    model.learn(total_timesteps=50000, callback=video_callback)
    
    model.save(f"{models_dir}/archery_ppo_fixed")
    print("Training Complete.")
    
    plot_results(log_dir)

    # --- Evaluation ---
    print("Visualizing performance...")
    env = ArcheryEnv(render_mode="human")
    obs, _ = env.reset()
    for _ in range(5):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"Shot Reward: {reward:.2f}")
        # Obs is reset automatically by the wrapper, but for clarity:
        if done: obs, _ = env.reset()

if __name__ == "__main__":
    train()
import time
from stable_baselines3 import PPO
from env import ArcheryEnv

def evaluate():
    model_path = "models/PPO/archery_ppo_fixed"
    env = ArcheryEnv(render_mode="human")
    
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print("Model not found! Run train.py first.")
        return

    print("Visualizing trained agent...")
    episodes = 10
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            
        print(f"Episode {ep+1} Score: {score:.2f}")
        time.sleep(0.5)

    env.close()

if __name__ == "__main__":
    evaluate()
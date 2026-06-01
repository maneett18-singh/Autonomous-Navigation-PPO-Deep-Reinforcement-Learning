import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import imageio

# Physics Constants
GRAVITY = 9.81
DT = 0.05
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_LEVEL = 50
METER_TO_PIXEL = 20.0

class ArcheryEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(ArcheryEnv, self).__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Action: [angle, power]
        # Angle: -1..1 (Mapped to 0..85 deg)
        # Power: 0..1 (Mapped to 0..35 m/s)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)

        # Observation: [target_x, target_y, dist_to_target, angle_to_target, wind]
        # We removed arrow info because the agent acts BEFORE the arrow flies.
        low = np.array([0, 0, 0, -np.pi, -10.0], dtype=np.float32)
        high = np.array([100, 50, 100, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_velocity = 35.0
        self.target_radius = 2.0 
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize target for robustness
        self.target_x = float(self.np_random.uniform(15.0, 35.0))
        self.target_y = float(self.np_random.uniform(2.0, 15.0))
        self.wind = float(self.np_random.uniform(-5.0, 5.0)) # Random wind

        # Default start pos
        self.arrow_x = 2.0
        self.arrow_y = 2.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.target_x - 2.0
        dy = self.target_y - 2.0
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        return np.array([self.target_x, self.target_y, dist, angle, self.wind], dtype=np.float32)

    def step(self, action):
        # 1. Decode Action
        angle_raw = np.clip(action[0], -1, 1)
        power_raw = np.clip(action[1], 0, 1)

        angle_deg = (angle_raw + 1) * 42.5 # 0 to 85 degrees
        angle_rad = math.radians(angle_deg)
        velocity = power_raw * self.max_velocity

        self.arrow_vx = velocity * math.cos(angle_rad)
        self.arrow_vy = velocity * math.sin(angle_rad)
        
        # 2. Simulate Flight (The "One-Shot" Loop)
        self.trajectory = [] # Store for visualization
        min_dist = float('inf')
        hit = False
        
        # We simulate up to 300 physics steps
        for _ in range(300):
            # Save for render
            self.trajectory.append((self.arrow_x, self.arrow_y))
            
            # Physics Update
            self.arrow_vx += self.wind * DT
            self.arrow_vy -= GRAVITY * DT
            
            new_x = self.arrow_x + self.arrow_vx * DT
            new_y = self.arrow_y + self.arrow_vy * DT
            
            # --- Better Hit Detection (Line Segment) ---
            # Checks if the arrow passed through the target in this step
            if self._check_segment_collision(self.arrow_x, self.arrow_y, new_x, new_y):
                hit = True
                min_dist = 0.0
                break # Stop simulation on hit
            
            # Track closest approach for reward shaping
            dist = math.sqrt((new_x - self.target_x)**2 + (new_y - self.target_y)**2)
            if dist < min_dist:
                min_dist = dist
            
            # Update Pos
            self.arrow_x, self.arrow_y = new_x, new_y
            
            # Stop if ground hit or OOB
            if self.arrow_y < 0 or self.arrow_x > 50:
                break

        # 3. Calculate Reward
        # Dense reward based on how close we ever got (min_dist)
        # This provides a smooth gradient even if we miss.
        reward = -min_dist 
        
        if hit:
            reward += 100.0 # Big bonus for actual hit
        
        # 4. Render if needed (Replay the shot)
        if self.render_mode == "human":
            self._render_trajectory(self.trajectory)

        # Episode is ALWAYS done after one shot
        return self._get_obs(), reward, True, False, {}

    def _check_segment_collision(self, x1, y1, x2, y2):
        """Checks if the line segment (x1,y1)->(x2,y2) intersects the target circle."""
        # Vector from p1 to p2
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return False

        # Project target onto line to find closest point
        t = ((self.target_x - x1) * dx + (self.target_y - y1) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t)) # Clamp to segment
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        dist_sq = (closest_x - self.target_x)**2 + (closest_y - self.target_y)**2
        return dist_sq <= self.target_radius**2

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Archery PPO")
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.screen.fill((135, 206, 235)) # Sky
        
        # Draw Ground
        pygame.draw.rect(self.screen, (34, 139, 34), (0, SCREEN_HEIGHT - GROUND_LEVEL, SCREEN_WIDTH, GROUND_LEVEL))

        # Target (Convert coords)
        tx = int(self.target_x * METER_TO_PIXEL)
        ty = int(SCREEN_HEIGHT - GROUND_LEVEL - (self.target_y * METER_TO_PIXEL))
        tr = int(self.target_radius * METER_TO_PIXEL)
        pygame.draw.circle(self.screen, (255, 0, 0), (tx, ty), tr)

        # Arrow
        sx = int(self.arrow_x * METER_TO_PIXEL)
        sy = int(SCREEN_HEIGHT - GROUND_LEVEL - (self.arrow_y * METER_TO_PIXEL))
        pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), 5)
        
        # For rgb_array
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _render_trajectory(self, trajectory):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        for (ax, ay) in trajectory:
            self.screen.fill((135, 206, 235))
            pygame.draw.rect(self.screen, (34, 139, 34), (0, SCREEN_HEIGHT - GROUND_LEVEL, SCREEN_WIDTH, GROUND_LEVEL))

            # Target
            tx = int(self.target_x * METER_TO_PIXEL)
            ty = int(SCREEN_HEIGHT - GROUND_LEVEL - (self.target_y * METER_TO_PIXEL))
            tr = int(self.target_radius * METER_TO_PIXEL)
            pygame.draw.circle(self.screen, (255, 0, 0), (tx, ty), tr)
            
            # Arrow
            sx = int(ax * METER_TO_PIXEL)
            sy = int(SCREEN_HEIGHT - GROUND_LEVEL - (ay * METER_TO_PIXEL))
            pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), 5)

            pygame.display.flip()
            self.clock.tick(60) # Fast playback

    def save_video(self, filepath):
        """Saves a video of the last trajectory."""
        if not hasattr(self, 'trajectory') or not self.trajectory:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            
        frames = []
        for (ax, ay) in self.trajectory:
            self.screen.fill((135, 206, 235))
            pygame.draw.rect(self.screen, (34, 139, 34), (0, SCREEN_HEIGHT - GROUND_LEVEL, SCREEN_WIDTH, GROUND_LEVEL))

            # Target
            tx = int(self.target_x * METER_TO_PIXEL)
            ty = int(SCREEN_HEIGHT - GROUND_LEVEL - (self.target_y * METER_TO_PIXEL))
            tr = int(self.target_radius * METER_TO_PIXEL)
            pygame.draw.circle(self.screen, (255, 0, 0), (tx, ty), tr)
            
            # Arrow
            sx = int(ax * METER_TO_PIXEL)
            sy = int(SCREEN_HEIGHT - GROUND_LEVEL - (ay * METER_TO_PIXEL))
            pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), 5)
            
            # Capture frame
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            frames.append(frame)
            
        # Save video
        imageio.mimsave(filepath, frames, fps=30)
        print(f"Video saved to {filepath}")

    def close(self):
        if self.screen: pygame.quit()
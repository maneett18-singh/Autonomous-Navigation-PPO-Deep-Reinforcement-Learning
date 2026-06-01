import pygame
import numpy as np
import time
import math
from env import ArcheryEnv
from utils import METER_TO_PIXEL, SCREEN_HEIGHT, GROUND_LEVEL

def manual_play():
    env = ArcheryEnv(render_mode="human")
    print("\n--- MANUAL CONTROL ---")
    print("W / S : Aim Up / Down")
    print("SPACE (Hold) : Charge Power")
    print("SPACE (Release): Fire")
    
    running = True
    while running:
        env.reset()
        env.render()
        done = False
        fired = False
        
        angle_input = 0.0 # -1 to 1 scale
        power_input = 0.0 # 0 to 1 scale
        charging = False
        
        while not done:
            # Event Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                
                # Input Handling
                if not fired:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        charging = True
                    if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                        charging = False
                        fired = True

            # Logic Update
            if not fired:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]: angle_input += 0.05
                if keys[pygame.K_s]: angle_input -= 0.05
                angle_input = np.clip(angle_input, -1, 1)

                if charging:
                    power_input += 0.02
                    power_input = min(power_input, 1.0)
                else:
                    # Decay power if not holding space
                    power_input = max(0.0, power_input - 0.05)

                # Just render aiming
                env.render()

                # Draw aiming direction
                aim_angle = (angle_input + 1) * 42.5
                aim_rad = math.radians(aim_angle)
                line_len = 4.0 + (power_input * 6.0) # Visual length varies with power
                
                ax = env.arrow_x + line_len * math.cos(aim_rad)
                ay = env.arrow_y + line_len * math.sin(aim_rad)
                
                start_screen_x = int(env.arrow_x * METER_TO_PIXEL)
                start_screen_y = int(SCREEN_HEIGHT - GROUND_LEVEL - (env.arrow_y * METER_TO_PIXEL))
                end_screen_x = int(ax * METER_TO_PIXEL)
                end_screen_y = int(SCREEN_HEIGHT - GROUND_LEVEL - (ay * METER_TO_PIXEL))
                
                pygame.draw.line(env.screen, (255, 0, 0), (start_screen_x, start_screen_y), (end_screen_x, end_screen_y), 3)
                
                # UI Overlay
                font = pygame.font.Font(None, 36)
                txt = font.render(f"Angle: {(angle_input+1)*42.5:.1f}  Power: {power_input*100:.0f}%", True, (0,0,0))
                env.screen.blit(txt, (10, 10))
                
                wind_txt = font.render(f"Wind: {env.wind:.2f} m/s", True, (0,0,255) if env.wind < 0 else (255,0,0))
                env.screen.blit(wind_txt, (10, 50))
                
                pygame.display.flip()
                env.clock.tick(30)
            
            else:
                # Agent (Physics) takes over
                action = np.array([angle_input, power_input], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if done:
                    outcome = "HIT!" if reward > 50 else "MISS"
                    print(f"Outcome: {outcome} | Reward: {reward:.2f}")
                    time.sleep(1)

        if not running: break

    env.close()

if __name__ == "__main__":
    manual_play()
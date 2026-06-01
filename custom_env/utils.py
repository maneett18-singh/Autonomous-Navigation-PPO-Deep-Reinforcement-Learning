import math

# --- Constants ---
GRAVITY = 9.81
DT = 0.05            # Time duration per step (seconds)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_LEVEL = 50    # Pixels from bottom
METER_TO_PIXEL = 20.0 # Scale: 1 meter = 20 pixels

def project_arrow(x, y, vx, vy):
    """
    Calculates the next position and velocity based on gravity.
    """
    new_x = x + vx * DT
    new_y = y + vy * DT
    new_vy = vy - GRAVITY * DT
    return new_x, new_y, vx, new_vy

def check_collision(arrow_x, arrow_y, target_x, target_y, target_radius):
    """
    Checks if the arrow coordinates are within the target radius.
    """
    dist = math.sqrt((arrow_x - target_x)**2 + (arrow_y - target_y)**2)
    return dist <= target_radius
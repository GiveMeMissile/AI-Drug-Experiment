from Visualization.visualization_functions import visual_loop
from Visualization import visual_constants as vc
from Environment.environment_functions import timestep
import pygame

HEADLESS = False

if __name__ == "__main__":
    if not HEADLESS:
        window = pygame.display.set_mode((vc.WINDOW_X, vc.WINDOW_Y))
        visual_loop(window)
    else:
        timestep()

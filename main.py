from Visualization.visualization_functions import visual_loop
from Visualization import visual_constants as vc
import pygame

if __name__ == "__main__":
    window = pygame.display.set_mode((vc.WINDOW_X, vc.WINDOW_Y))
    visual_loop(window)

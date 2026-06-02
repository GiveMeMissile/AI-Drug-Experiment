import pygame
from Visualization import visual_constants as vc

class GridObject:
    def __init__(self, dim, x, y, thickness):

        # Save Important variables ig
        self.dim = dim
        self.x = x
        self.y = y
        self.thickness = thickness

        # Create the two squares
        self.gray_square = pygame.Rect(x, y, dim, dim)
        self.black_square = pygame.Rect(x + thickness, y + thickness, dim - thickness*2, dim - thickness*2)

    def render(self, window):
        pygame.draw.rect(window, vc.GRAY, self.gray_square)
        pygame.draw.rect(window, vc.BLACK, self.black_square)

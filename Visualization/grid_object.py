from Visualization import visual_constants as vc
from Environment import objects as obj
import pygame


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

        # Variables for the Display Square
        self.display_square = None
        self.display_obj = None
        self.display_dim = None

    def render(self, window):
        pygame.draw.rect(window, vc.GRAY, self.gray_square)
        pygame.draw.rect(window, vc.BLACK, self.black_square)
        if self.display_square is not None:
            pygame.draw.rect(window, (self.display_obj.get_red(), self.display_obj.get_green(), self.display_obj.get_blue()), self.display_square)
    
    def remove_display(self):
        self.display_square = None
        self.display_obj = None
        self.display_dim = None

    def set_display_square(self, object):
        self.display_obj = object

        if isinstance(object, obj.Food):
            self.display_dim = vc.FOOD_DIM
        if isinstance(object, obj.Drug):
            self.display_dim = vc.DRUG_DIM
        if isinstance(object, obj.AI_Agent):
            self.display_dim = vc.AI_DIM

        location_change = (self.dim - self.display_dim)/2
        self.display_square = pygame.Rect(self.x + location_change, self.y + location_change, self.display_dim, self.display_dim)
        
from Visualization.visualization_functions import visual_loop
from Environment.environment_functions import headless_loop
from Visualization import visual_constants as vc
from AIManagement.ai_manager import AIManager
import pygame

HEADLESS = False
NUMBER_OF_LOOPS = 10
TRAINING = True

if __name__ == "__main__":
    ai_manager = AIManager()

    for i in range(NUMBER_OF_LOOPS):

        if not HEADLESS:
            window = pygame.display.set_mode((vc.WINDOW_X, vc.WINDOW_Y))
            end = visual_loop(window, ai_manager, TRAINING)
            if end:
                break
        else:
            headless_loop(ai_manager, TRAINING)
        ai_manager.end_loop()
    
    ai_manager.save_model()
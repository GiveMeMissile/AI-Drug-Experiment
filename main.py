from Visualization.visualization_functions import visual_loop
from Environment.environment_functions import headless_loop
from AIManagement.ai_functions import check_for_folder
from Visualization import visual_constants as vc
from AIManagement.ai_manager import AIManager
import pygame

# Important settings
HEADLESS = False
NUMBER_OF_LOOPS = 120
TRAINING = False
ITERATION = 4  # Testing lol

if __name__ == "__main__":
    check_for_folder()
    ai_manager = AIManager(ITERATION)

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
    ai_manager.save_tracking_data()
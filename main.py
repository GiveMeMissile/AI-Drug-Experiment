from Visualization.visualization_functions import visual_loop
from Environment.environment_functions import headless_loop
from AIManagement.ai_functions import check_for_folder
from Visualization import visual_constants as vc
from AIManagement.ai_manager import AIManager
import pygame

# AI Training Notes:
#  - Increased training time, 120 episodes, seems to work very well. +
#  - Changes in food represntation as provided a nice improvement. +
#  - Descreasing Discount Factor did not fix indecision issue, and even make some of it worse. -
#  - Making the Model bigger did not have any major increase on the preformance of the model. 0
#  - Strange results on the parabolification of the rewards, AI seems to act with more vigor, though Action veriety went down
#      as loss went up, IDK what to make of this change yet... 0
#  - The Wall detrement worked splendedly, the AI no longer rams itself endlessly into the wall, instead it ocillates -_- +
#  - DON'T DECREASE THE LEARNING RATE, THE AI WILL SUCK -
#  - Don't increase Learning rate either - 

# Important settings
HEADLESS = False
NUMBER_OF_LOOPS = 120
TRAINING = False
ITERATION = 1 # Testing lol
IS_LSTM = False

if __name__ == "__main__":
    check_for_folder()
    ai_manager = AIManager(ITERATION, IS_LSTM)

    num_loops = 0

    for i in range(NUMBER_OF_LOOPS):

        if not HEADLESS:
            window = pygame.display.set_mode((vc.WINDOW_X, vc.WINDOW_Y))
            end = visual_loop(window, ai_manager, TRAINING)
            if end:
                break
        else:
            headless_loop(ai_manager, TRAINING)
        ai_manager.end_loop()
        num_loops += 1
        print(f"Loop #{num_loops}  |  Percentage towards completion: {100*(num_loops/NUMBER_OF_LOOPS)}%")
    
    ai_manager.save_model()
    ai_manager.save_tracking_data()
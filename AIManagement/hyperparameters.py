import Environment.environmental_constants as ec

# AI Hyperparameters
NUM_LAYERS = 2
HIDDEN_SIZE = 64
INPUT_SIZE = 4 + 5*(ec.MAX_NUM_OBJECTS - 1)
OUTPUT_SIZE = 4
LEARNING_RATE = 0.0001 
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
EPSILON_DECAY = 0.01

# AI Saving Information
MODEL_SAVE_FOLDER = "AIManagement/models"
MODEL_INFO = MODEL_SAVE_FOLDER + "/model_info.json"
MODEL_DIR = MODEL_SAVE_FOLDER + "/model"
INFO_FORMAT = {"model number": [], "LSTM": [], "hidden": [], "layers": [], "input": [], "epsilon": [], "iteration": []}

# Progress Tracking Info
DATA_SAVE_FOLDER = "AIManagement/progress"
DATA_SAVE_DIR = DATA_SAVE_FOLDER + "/save_"
DATA_SAVE_INFO = DATA_SAVE_FOLDER + "/save_info.json"
SAVE_FORMAT = {"save": [], "model": []}

# Other Constants
MAX_SAVED_EPISODES = 10
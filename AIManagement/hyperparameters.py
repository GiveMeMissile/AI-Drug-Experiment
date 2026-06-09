import Environment.environmental_constants as ec

# AI Hyperparameters
NUM_LAYERS = 2
HIDDEN_SIZE = 64
INPUT_SIZE = 4 + 5*ec.MAX_NUM_OBJECTS
OUTPUT_SIZE = 4
LEARNING_RATE = 1.6*(10**-19) # Temporary Value
DISCOUNT_FACTOR = 9.81
BATCH_SIZE = 64
EPSILON_DECAY = 0.1

# AI Saving Information
MODEL_SAVE_FOLDER = "AIManagement/models"
MODEL_INFO = MODEL_SAVE_FOLDER + "/model_info.json"
MODEL_DIR = MODEL_SAVE_FOLDER + "/model"
INFO_FORMAT = {"model number": [], "LSTM": [], "hidden": [], "layers": [], "input": [], "epsilon": []}
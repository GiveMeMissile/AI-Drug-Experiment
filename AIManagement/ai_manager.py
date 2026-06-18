from AIManagement.data_tracker import TrainingData
from AIManagement.ai_functions import get_lowest
from AIManagement import neural_networks as nns
from AIManagement import hyperparameters as hp
from Environment.objects import AI_Agent
from torch import nn
import torch
import json

class AIManager:

    info = {}
    model_number = -1
    epsilon = 1

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ended = False

        # Load models
        self.policy_model = nns.SimpleNN().to(self.device)
        self.load_model()
        self.target_model = nns.SimpleNN().to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), hp.LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        # Important Data
        self.previous_state = None
        self.action = None

        # Data saving
        self.training_data = TrainingData(5)

    def set_agent(self, agent):
        self.agent = agent
        
    def end_loop(self):
        self.epsilon -= hp.EPSILON_DECAY
        self.agent = None
        self.ended = False

    def create_input(self, objects):
        input_list = [len(objects) - 1, self.agent.x, self.agent.y, self.agent.initial_value]

        # Add all objects to the input
        for obj in objects:
            if isinstance(obj, AI_Agent):
                continue
            input_list.append(obj.x)
            input_list.append(obj.y)
            input_list.append(obj.initial_value)
            input_list.append(obj.rate_of_decay)
            input_list.append(obj.num_timesteps)

        # Add padding values for non existant objects (padding = -1)
        for _ in range(hp.INPUT_SIZE - len(input_list)):
            input_list.append(-1)

        return torch.tensor(input_list).to(self.device)
    
    def calculate_reward(self):
        # Calculates the reward of the AI.

        reward = self.agent.initial_value
        return reward
    
    def save_data(self, current_state):

        if self.previous_state is None:
            return

        ended = False
        if self.agent.initial_value < -1:
            ended = True
            self.ended = True
        
        self.training_data.save_data(
            self.previous_state,
            self.action,
            current_state,
            self.calculate_reward(),
            ended
        )
    
    def get_q_values(self, objects, action=None):
        input_tensor = self.create_input(objects)

        self.save_data(input_tensor)
        self.previous_state = input_tensor

        q_values = self.policy_model(input_tensor)

        if action is None:
            self.action = q_values.argmax()
        else:
            self.action = action

        q_values = q_values.detach().tolist()
        return q_values
    
    def train(self, sample=None):
        # Trains the simple NN based off of the saved data in the DataTracker().
        # Note: The parameter sample is meant for testing purposes

        # Check the data sample to ensure that it is not None
        if sample is None:
            sample = self.training_data.get_sample()
        if sample is None:
            return None

        # Organize sample data into batched seperate parts
        batch_size = len(sample)
        previous_states = torch.stack([sample[i][0] for i in range(batch_size)], dim=0)
        actions = torch.tensor([sample[i][1] for i in range(batch_size)]).to(self.device).unsqueeze(1)
        new_states = torch.stack([sample[i][2] for i in range(batch_size)], dim=0)
        rewards = torch.tensor([sample[i][3] for i in range(batch_size)]).to(self.device)
        ended = [sample[i][4] for i in range(batch_size)]

        q_values = self.policy_model(previous_states)

        # Calculate the target value
        with torch.no_grad():
            target_q_values = self.target_model(new_states).max(1)
            if not ended:
                target = rewards + target_q_values * hp.DISCOUNT_FACTOR
            else:
                target = rewards
        
        # print(f"Q Values: {q_values}")
        # print(f"Actions: {actions}")
        # print(f"Targets: {target}")

        
        q_values = q_values.gather(1, actions)
        # print(q_values)
        loss = self.loss_fn(q_values, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()
    
    def find_model(self):
        # Finds a model to be loaded

        with open(hp.MODEL_INFO) as f:
            self.info = json.load(f)
        
        for i in range(len(self.info["model number"])):
            if (self.info["hidden"][i] == hp.HIDDEN_SIZE and self.info["layers"][i] == hp.NUM_LAYERS and self.info["input"][i] == hp.INPUT_SIZE):
                self.model_number = self.info["model number"][i]
                self.epsilon = self.info["epsilon"][i]
                break

    def load_model(self):
        # Loads a saved model from the saved models based off of the current hyperparams

        self.find_model()
        if self.model_number == -1:
            return
        
        self.policy_model.load_state_dict(torch.load(hp.MODEL_DIR + "_" + str(self.model_number) + ".pth"))

    def save_model(self):
        # Saves the AI model a file as well as its metadata in the info filw

        if self.model_number == -1:
            self.model_number = get_lowest(self.info["model number"])
            self.info["model number"].append(self.model_number)
            self.info["LSTM"].append(False)  # Temporary
            self.info["hidden"].append(hp.HIDDEN_SIZE)
            self.info["layers"].append(hp.NUM_LAYERS)
            self.info["input"].append(hp.INPUT_SIZE)
            self.info["epsilon"].append(self.epsilon)
        else:
            idx = -1
            for i in range(len(self.info["model number"])):
                if self.info["model number"][i] == self.model_number:
                    idx = i
                    break
            self.info["epsilon"][idx] = self.epsilon

        with open(hp.MODEL_INFO, "w") as f:
            json.dump(self.info, f)

        torch.save(self.policy_model.state_dict(), hp.MODEL_DIR + "_" + str(self.model_number) + ".pth")
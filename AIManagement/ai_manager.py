from AIManagement.data_tracker import TrainingData, DataMonitor
from AIManagement.ai_functions import get_lowest
from AIManagement import neural_networks as nns
from AIManagement import hyperparameters as hp
from Environment.objects import AI_Agent
from torch import nn
import torch
import json
import math

class AIManager:

    info = {}
    model_number = -1
    epsilon = 1

    def __init__(self, iteration):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ended = False
        self.iteration = iteration

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
        self.step = 0
        self.q_values = None
        self.loss = None
        self.contacted_obj = False
        self.reward = None

        # Data saving
        self.training_data = TrainingData(hp.MAX_SAVED_EPISODES)
        self.tracking_data = DataMonitor(self.model_number)

    def set_agent(self, agent):
        self.agent = agent

    def sync_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
        
    def end_loop(self):
        self.epsilon -= hp.EPSILON_DECAY
        self.agent = None
        self.ended = False
        self.training_data.new_episode()
        self.tracking_data.new_episode()

    def order_objects(self, objs):
        # Orders the list of objects based on their distance from the AI

        order_list = []
        for i, obj in enumerate(objs):
            a = math.sqrt((obj[0])**2 + (obj[1])**2)
            order_list.append((a, i))
        order_list.sort(key=lambda x: x[0])
        ordered_list = []
        for part in order_list:
            ordered_list.append(objs[part[1]])
        return ordered_list

    def create_input(self, objects):
        input_list = [len(objects) - 1, self.agent.x, self.agent.y, self.agent.initial_value]

        obj_list = []

        # Add all objects to the input
        for obj in objects:
            if isinstance(obj, AI_Agent):
                continue
            a = []
            a.append(obj.x - self.agent.x)
            a.append(obj.y - self.agent.y)
            a.append(obj.initial_value)
            a.append(obj.rate_of_decay)
            obj_list.append(a)
            a.append(obj.num_timesteps)

        ordered_objects = self.order_objects(obj_list)
        # ordered_objects = obj_list
        for obj in ordered_objects:
            input_list += obj

        # Add padding values for non existant objects (padding = 0)
        for _ in range(hp.INPUT_SIZE - len(input_list)):
            input_list.append(0)

        return torch.tensor(input_list).to(self.device)
    
    def calculate_reward(self):
        # Calculates the reward of the AI.

        reward = ((2*self.agent.initial_value)**2) * self.agent.initial_value/abs(self.agent.initial_value)

        # Bonus for consuming an object
        if self.agent.object_consumed is not None:
            self.contacted_obj = True
            reward += self.agent.object_consumed.calculate_current_value()
        else:
            self.contacted_obj = False

        if self.agent.wall:
            reward -= 2

        self.reward = reward
        return reward
    
    def save_data(self, current_state):

        if self.previous_state is None:
            return

        ended = False
        if self.agent.initial_value <= -1:
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
            self.action = q_values.argmax().detach().item()
        else:
            self.action = action

        q_values = q_values.detach().tolist()
        self.q_values = q_values
        self.step += 1
        return q_values
    
    def train(self, sample=None):
        # Trains the simple NN based off of the saved data in the DataTracker().
        # Note: The parameter sample is meant for testing purposes

        # Check the data sample to ensure that it is not None
        self.loss = None
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
        
        # Debug Code
        # print(f"Q Values: {q_values}")
        # print(f"Actions: {actions}")
        # print(f"Targets: {target}")

        q_values = q_values.gather(1, actions)
        loss = self.loss_fn(q_values, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.detach().item()

        return self.loss
    
    def find_model(self):
        # Finds a model to be loaded

        with open(hp.MODEL_INFO) as f:
            self.info = json.load(f)
        
        for i in range(len(self.info["model number"])):
            if (self.info["hidden"][i] == hp.HIDDEN_SIZE and self.info["layers"][i] == hp.NUM_LAYERS and 
                self.info["input"][i] == hp.INPUT_SIZE and self.info["iteration"][i] == self.iteration):
                self.model_number = self.info["model number"][i]
                self.epsilon = self.info["epsilon"][i]
                break

    def track_data(self):
        self.tracking_data.add_data(self.step, self.q_values, self.action, self.reward, self.contacted_obj, self.loss)

    def load_model(self):
        # Loads a saved model from the saved models based off of the current hyperparams

        self.find_model()
        if self.model_number == -1:
            print("Creating a new Model")
            return
        print(f"Loading Model #{self.model_number}")
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
            self.info["iteration"].append(self.iteration)
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

    def save_tracking_data(self):
        self.tracking_data.save_data(self.model_number)
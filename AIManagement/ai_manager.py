from AIManagement.data_tracker import TrainingData
from AIManagement import neural_networks as nns
from AIManagement import hyperparameters as hp
from Environment.objects import AI_Agent
import torch.nn.functional as f
import torch

class AIManager:

    def __init__(self, agent):
        self.agent = agent
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models
        self.policy_model = nns.SimpleNN().to(self.device)
        self.target_model = nns.SimpleNN().to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), hp.LEARNING_RATE)
        self.loss_fn = f.smooth_l1_loss()

        # Important Data
        self.previous_state = None
        self.action = None

        # Data saving
        self.training_data = TrainingData(5)

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
        reward = 2*(reward - 0.5)
        return reward
    
    def save_data(self, current_state):

        if self.previous_state is None:
            return

        ended = False
        if self.agent.initial_value < 0:
            ended = True
        
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
    
    def train(self):
        sample = self.training_data.get_sample()

        # Organize sample data into batched seperate parts
        batch_size = len(sample)
        previous_states = torch.stack([sample[i][0] for i in range(batch_size)], dim=0)
        actions = torch.tensor([sample[i][1] for i in range(batch_size)])
        new_states = torch.stack([sample[i][2] for i in range(batch_size)], dim=0)
        rewards = torch.tensor([sample[i][3] for i in range(batch_size)])
        ended = [sample[i][4] for i in range(batch_size)]

        q_values = self.policy_model(previous_states)

        # Calculate the target value
        with torch.no_grad():
            target_q_values = self.target_model(new_states).max(1)
            if not ended:
                target = rewards + target_q_values * hp.DISCOUNT_FACTOR
            else:
                target = rewards
        
        q_values = q_values.gather(1, actions)
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()
        


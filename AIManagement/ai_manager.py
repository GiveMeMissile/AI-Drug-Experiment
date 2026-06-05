from AIManagement import neural_networks as nns
from AIManagement import hyperparameters as hp
from Environment.objects import AI_Agent
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
    
    def get_q_values(self, objects):
        input_tensor = self.create_input(objects)

        q_values = self.policy_model(input_tensor)
        q_values = q_values.detach().tolist()
        return q_values


from Environment import environmental_constants as ec
from AIManagement import hyperparameters as hp
import random
import math

class Object:
    
    def __init__(self, x, y, rate_of_decay, initial_value):
        self.x = x
        self.y = y
        self.rate_of_decay = rate_of_decay
        self.initial_value = initial_value
        self.num_timesteps = 0
        self.contacted_object = False
        self.wall = False
    
    def update_location(self, x, y):
        self.x = x
        self.y = y

    def count_timestep(self):
        # Counts the timesteps based on when it is called

        self.num_timesteps += 1

    def get_green(self):
        # Calculates amount of green of the rgb based on the object's variables.

        g = 255/2 + (self.initial_value - self.rate_of_decay*self.num_timesteps)*(255/2)
        return g
    
    def get_blue(self):
        # Calculates the amount of blue of the rgb based on the object's variables.

        b = (1 - 10*(self.rate_of_decay))*(255)
        return b
    
    def get_red(self):
        # Calculates the red in rgb, unused for all Objects except AI_Agent

        return 0
    
    def apply_effect(self, obj):
        # Function to be overridden by other Objects
        return 0

class Food(Object):
    
    def __init__(self, x, y):
        super().__init__(x, y,  random.randint(ec.MIN_DECAY, ec.MAX_DECAY)/100, random.randint(ec.MIN_INITIAL, ec.MAX_INITIAL)/100)

    def calculate_current_value(self):
        # Calcululates by how much the food has decayed based on how many timesteps have occured.

        return self.initial_value - self.rate_of_decay*self.num_timesteps
    
    def apply_effect(self, obj):
        # Gets the effects of what happens when the AI_Agent object comes into contact with the food.
        
        obj.initial_value += self.calculate_current_value()
        if obj.initial_value > 1:
            obj.initial_value = 1
        obj.object_consumed = self


class AI_Agent(Object):
    
    def __init__(self, x, y, grid, manager):
        super().__init__(x, y, ec.AGENT_DECAY, 1)
        self.grid = grid
        self.manager = manager
        self.epsilon = manager.epsilon
        self.object_consumed = None
        self.addiction = 0
        self.drug_reward = 0
        self.initial_drug_reward = 0
        self.time_since_drug = 0
        self.minimum_drug_reward = 0

    def recalculate_drugs(self):
        self.time_since_drug += 1
        if self.drug_reward <= self.minimum_drug_reward:
            self.drug_reward = self.minimum_drug_reward
        else:
            self.drug_reward = self.initial_drug_reward - round((3/8)*((self.time_since_drug)**2), 2)
            # print(f"Drug Reward: {self.drug_reward}  |  Addiction: {self.addiction}  |  Minimum: {self.minimum_drug_reward}")

    def count_timestep(self):
        super().count_timestep()
        self.initial_value -= self.rate_of_decay
        self.recalculate_drugs()

    def reset(self):
        self.initial_value = 1
        self.epsilon -= hp.EPSILON_DECAY

    def move(self, objs):
        # Moves the AI to a different location

        value = None
        if self.epsilon > random.random():
            value = random.randint(0, 3)
        q_values = self.manager.get_q_values(objs, action=value)
        if value is None:
            value = q_values.index(max(q_values))
        old_x = self.x
        old_y = self.y

        if value == 0:
            self.y += 1
        elif value == 1:
            self.y -= 1
        elif value == 2:
            self.x += 1
        elif value == 3:
            self.x -= 1
            
        self.object_consumed = None
        self.grid.move_object(old_x, old_y, self.x, self.y)
    
    def get_blue(self):
        return 0
    
    def get_red(self):
        return 255
    
    def get_green(self):
        # Calculates amount of green of the rgb based on the object's variables.

        g = 255/2 + (self.initial_value)*(255/2)
        return g
    

class Drug(Object):
    # FINALLY I CAN CREATE THIS GLORIOUS THING!!!

    def __init__(self, x, y):
        super().__init__(x, y, 0, random.randint(ec.DRUG_MIN_INITIAL, ec.DRUG_MAX_INITIAL)/100)

    def apply_effect(self, obj):
        if isinstance(obj, AI_Agent):
            # print("Drug Consumed!")
            obj.addiction += ec.ADDICTION_INCREASE
            obj.drug_reward += (ec.MIN_DRUG_REWARD + abs(self.initial_value) * 5) * (1 - obj.addiction)
            obj.initial_drug_reward = obj.drug_reward
            obj.time_since_drug = 0
            obj.minimum_drug_reward = -round(2*math.sqrt(5*obj.addiction), 1)

        obj.initial_value += self.initial_value
        obj.contacted_object = self
    
    def get_red(self):
        return 230

    
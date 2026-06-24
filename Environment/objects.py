from Environment import environmental_constants as ec
from AIManagement import hyperparameters as hp
import random

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

        b = (1 - 5*(self.rate_of_decay))*(255)
        return b
    
    def get_red(self):
        # Calculates the red in rgb, unused for all Objects except AI_Agent

        return 0
    
    def apply_effect(self, obj):
        # Function to be overridden by other Objects
        return 0

class Food(Object):
    
    def __init__(self, x, y, rate_of_decay, initial_value):
        super().__init__(x, y, rate_of_decay, initial_value)

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

    def count_timestep(self):
        super().count_timestep()
        self.initial_value -= self.rate_of_decay

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

    def get_green(self):
        # No Green for Drugs
        return 0
from Environment import environmental_constants as ec
import random

class Object:
    
    def __init__(self, x, y, rate_of_decay, initial_value):
        self.x = x
        self.y = y
        self.rate_of_decay = rate_of_decay
        self.initial_value = initial_value
        self.num_timesteps = 0
    
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
    
    def get_object_effects(self):
        # Function to be overridden by other Objects

        pass

class Food(Object):
    
    def __init__(self, x, y, rate_of_decay=random.randint(5, 10)/100, initial_value=random.randint(50, 70)/100):
        super().__init__(x, y, rate_of_decay, initial_value)

    def calculate_current_value(self):
        # Calcululates by how much the food has decayed based on how many timesteps have occured.

        return self.initial_value - self.rate_of_decay*self.num_timesteps
    
    def get_object_effects(self):
        # Gets the effects of what happens when the AI_Agent object comes into contact with the food.
        # This has yet to be set up.
        return 0


class AI_Agent(Object):
    
    def __init__(self, x, y, grid):
        super().__init__(x, y, ec.AGENT_DECAY , 1)
        self.grid = grid
        self.manager = None

    def count_timestep(self):
        super().count_timestep()
        self.initial_value -= self.rate_of_decay

    def set_manager(self, manager):
        self.manager = manager

    def move(self, objs):
        # Moves the AI to a different location

        # Simulates AI choice
        # direction = [random.random(), random.random(), random.random(), random.random()]
        q_values = self.manager.get_q_values(objs)
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

        self.grid.move_object(old_x, old_y, self.x, self.y)
    
    def get_blue(self):
        return 0
    
    def get_red(self):
        return 255
    

class Drug(Object):

    def get_green(self):
        # No Green for Drugs
        return 0
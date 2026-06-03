class Object:
    
    def __init__(self, x, y, rate_of_decay, initial_value):
        self.x = x
        self.y = y
        self.rate_of_decay = rate_of_decay
        self.initial_value = initial_value
        self.num_timesteps = 0

class Food(Object):
    pass

class AI_Agent(Object):
    pass

class Drug(Object):
    pass
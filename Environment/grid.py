import Environment.objects as objs

class Grid:

    def __init__(self, num_rows, num_columns):

        # Create Grid
        self.grid = []
        for _ in range(num_rows):
            g = []
            for _ in range(num_columns):
                g.append(None)
            self.grid.append(g)

        # Save other important variables
        self.num_objects = 0
        self.num_rows = num_rows
        self.num_columns = num_columns

    def adjust_positions(self, x, y):
        # Adjusts the inputted position values if they are out of the range of the grid

        adjusted = False

        if x > self.num_columns - 1:
            x = self.num_columns - 1
            adjusted = True
        elif x < 0:
            x = 0
            adjusted = True

        if y > self.num_rows - 1:
            y = self.num_rows - 1
            adjusted = True
        elif y < 0:
            y = 0
            adjusted = True
        
        return x, y, adjusted

    def add_object(self, x, y, obj):
        # Adds an object to the grid environment
        x, y, _ = self.adjust_positions(x, y)

        self.num_objects += 1
        self.grid[y][x] = obj

    def remove_object(self, x, y):
        # Removed an object from the grid enviornment

        new_x, new_y, _ = self.adjust_positions(x, y)
        if (not new_x == x) or (not new_y == y):
            return False

        if self.grid[y][x] is None:
            return False
        
        self.num_objects -= 1
        self.grid[y][x] = None
        return True
    
    def move_object(self, old_x, old_y, x, y):
        # Moves an object from one position to another

        new_x, new_y, _ = self.adjust_positions(old_x, old_y)
        if (not new_x == old_x) or (not new_y == old_y):
            return False

        obj = self.grid[old_y][old_x]

        x, y, wall = self.adjust_positions(x, y)
        self.grid[old_y][old_x].wall = wall

        if self.object_exists_at(x, y):
            self.grid[y][x].apply_effect(obj)
            self.remove_object(x, y)


        self.remove_object(old_x, old_y)
        self.add_object(x, y, obj)

    def object_exists_at(self, x, y):
        # Checks if an object exists at the inputted location of the grid.

        x, y, _ = self.adjust_positions(x, y)
        if isinstance(self.grid[y][x], objs.Object):
            return True
        return False
    
    def get_all_objects(self):
        all_objs = []
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                if isinstance(self.grid[i][j], objs.Object):
                    all_objs.append((self.grid[i][j], j, i))
        
        return all_objs
    
    def is_object_in_grid(self, obj):
        for row in self.grid:
            if obj in row:
                return True
        return False
    
    def how_many_objs(self, obj_type):
        num_objs = 0

        for row in self.grid:
            for space in row:
                if isinstance(space, obj_type):
                    num_objs += 1
        return num_objs
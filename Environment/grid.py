class Grid:
    def __init__(self, num_rows, num_columns):
        self.grid = []
        for i in range(num_rows):
            g = []
            for _ in range(num_columns):
                g.append(None)
            self.grid.append(g)
        self.num_objects = 0
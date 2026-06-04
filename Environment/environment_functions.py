import Environment.environmental_constants as ev
import AIManagement.neural_networks as nns
from Environment import objects as obj
from Environment.grid import Grid
import random


def headless_loop():
    grid = Grid(ev.NUM_ROWS, ev.NUM_COLUMNS)

    # More Test code
    while True:
        timestep(grid)
        end = input("Type NO to stop: ")
        if end == "NO":
            break


def timestep(grid):
    # Temporary test code testing spawn_foods

    spawn_foods(grid)
    print("*******************************************************************")
    for i in range(grid.num_rows):
        for j in range(grid.num_columns):
            if grid.object_exists_at(j, i):
                print(f"Food exists at ({j}, {i})")
                print(f"Object is {grid.grid[i][j].__class__}")
    print("*******************************************************************\n")


def spawn_foods(grid):
    # Has a 1/4 chance to spawn a food object for every object which does not exist

    max_spawn = ev.MAX_NUM_OBJECTS - grid.num_objects
    for _ in range(max_spawn):
        if (random.choice([True, True, True, False])):
            continue
        while True:
            x = random.randint(0, grid.num_columns - 1)
            y = random.randint(0, grid.num_rows - 1)
            if grid.object_exists_at(x, y):
                continue
            grid.add_object(x, y, obj.Food(x, y))
            break


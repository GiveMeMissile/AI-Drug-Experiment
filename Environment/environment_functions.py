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
    # Runs a full timestep of the 

    spawn_foods(grid)
    all_objs = grid.get_all_objects()
    only_obj = []
    for obj_info in all_objs:
        only_obj.append(obj_info[0])

    for obj_info in all_objs:
        obj_info[0].update_location(obj_info[1], obj_info[2])
        obj_info[0].count_timestep()

        if isinstance(obj_info[0], obj.Food):
            value = obj_info[0].calculate_current_value()
            if value < -1:
                grid.remove_object(obj_info[1], obj_info[2])

        if isinstance(obj_info[0], obj.AI_Agent):
            if obj_info[0].initial_value < 0:
                grid.remove_object(obj_info[1], obj_info[2])
            obj_info[0].move(only_obj)


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


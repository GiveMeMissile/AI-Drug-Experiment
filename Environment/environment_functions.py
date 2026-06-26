import Environment.environmental_constants as ec
from Environment import objects as obj
from Environment.grid import Grid
import random


def headless_loop(ai_manager, train):
    grid = Grid(ec.NUM_ROWS, ec.NUM_COLUMNS)
    agent = obj.AI_Agent(random.randint(0, ec.NUM_COLUMNS), random.randint(0, ec.NUM_ROWS), grid, ai_manager)
    grid.add_object(random.randint(0, ec.NUM_COLUMNS), random.randint(0, ec.NUM_ROWS), agent)
    ai_manager.set_agent(agent)

    num_time_steps = 0
    while num_time_steps < ec.MAX_TIME_STEPS:
        timestep(grid, ai_manager)
        if agent.initial_value < -1:
            break
        num_time_steps += 1
        if num_time_steps % ec.STEPS_TILL_SYNC == 0:
            ai_manager.sync_model()

        if train:
            ai_manager.train()
        ai_manager.track_data()
        if ai_manager.ended:
            break
        if not grid.is_object_in_grid(agent):
            ai_manager.end([an_obj[0] for an_obj in grid.get_all_objects()])
            break

        # Debug code
        # if num_time_steps == ec.MAX_TIME_STEPS:
        #     print(f"Ended: {ai_manager.ended}  |  Agent : {agent.initial_value}  |  Exists: {grid.is_object_in_grid(agent)}")


def timestep(grid, manager):
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
            obj_info[0].move(only_obj)
            if obj_info[0].initial_value < -1:
                grid.remove_object(obj_info[0].x, obj_info[0].y)
            if not manager.ended:
                manager.end(only_obj)
            # print(f"Initial Value: {obj_info[0].initial_value}  |  Red: {obj_info[0].get_red()}  |  Green: {obj_info[0].get_green()}  |  Blue: {obj_info[0].get_blue()}")


def spawn_foods(grid):
    # Has a 1/4 chance to spawn a food object for every object which does not exist

    max_spawn = ec.MAX_NUM_OBJECTS - grid.num_objects
    for _ in range(max_spawn):
        if (random.choice([True, True, True, False])):
            continue
        while True:
            x = random.randint(0, grid.num_columns - 1)
            y = random.randint(0, grid.num_rows - 1)
            if grid.object_exists_at(x, y):
                continue
            grid.add_object(x, y, obj.Food(x, y, random.randint(ec.MIN_DECAY, ec.MAX_DECAY)/100, random.randint(ec.MIN_INITIAL, ec.MAX_INITIAL)/100))
            break


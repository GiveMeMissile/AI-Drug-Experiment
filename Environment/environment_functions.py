import AIManagement.neural_networks as nns
import Environment.environmental_constants as ev
from Environment.grid import Grid
from Environment import objects as obj

def timestep():
    # Temporary test code

    grid = Grid(ev.NUM_ROWS, ev.NUM_COLUMNS)
    object = obj.Object(2, 2, 0, 0)
    grid.add_object(2, 2, object)
    print(f"Grid: {grid.grid}")
    print(f"Object at (2, 2): {grid.object_exists_at(2, 2)}")
    print(f"Object at (1, 9): {grid.object_exists_at(1, 9)}")
    grid.remove_object(2, 2)
    print(f"Object at (2, 2) after removal: {grid.object_exists_at(2, 2)}")
    grid.add_object(24, 8, object)
    print(f"Object at (19, 8): {grid.object_exists_at(19, 8)}")
    grid.move_object(19, 8, 4, 4)
    print(f"Object at (19, 8) After Move: {grid.object_exists_at(19, 8)}")
    print(f"Object at (4, 4) After Move: {grid.object_exists_at(4, 4)}")


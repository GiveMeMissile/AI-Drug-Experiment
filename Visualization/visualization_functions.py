from Environment.environment_functions import timestep
from Environment import environmental_constants as ec
from Visualization import visual_constants as vc
from Visualization import grid_object
from Environment.grid import Grid
import pygame


def visual_loop(window):
    pygame.init()
    running = True
    clock = pygame.time.Clock()
    visual_grid = create_grid()
    grid = Grid(ec.NUM_ROWS, ec.NUM_COLUMNS)
    delay = pygame.time.get_ticks()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if pygame.time.get_ticks() - delay >= vc.TIME_PER_TIMESTEP:           
            delay = pygame.time.get_ticks()
            timestep(grid)

            # Connecting what is on the real grid to the visual grid
            visual_objects = get_grid_objects(visual_grid)
            actual_objects = grid.get_all_objects()
            for location in visual_objects:
                for object in actual_objects:
                    if object[1] == location[0] and object[2] == location[1]:
                        visual_objects.remove(location)
                        actual_objects.remove(object)
                        break
            
            for location in visual_objects:
                visual_grid[location[1]][location[0]].remove_display()

            for object in actual_objects:
                visual_grid[object[2]][object[1]].set_display_square(object[0])

        draw(window, visual_grid)
        clock.tick(60)


def get_grid_objects(visual_grid):
    locations = []
    for i in range(len(visual_grid)):
        for j in range(len(visual_grid[i])):
            if visual_grid[i][j].display_square is not None:
                locations.append((j, i))
    
    return locations


def create_grid():
    x = 0
    y = 100
    grid = []
    for _ in range(10):
        row = []
        x = 0
        for _ in range(20):
            row.append(grid_object.GridObject(vc.GRID_BOX_DIM, x, y, vc.GRID_THICKNESS))
            x += vc.GRID_BOX_DIM - vc.GRID_THICKNESS
        grid.append(row)
        y += vc.GRID_BOX_DIM - vc.GRID_THICKNESS
    return grid


def draw(window, grid):
    window.fill(vc.BLACK)
    for row in grid:
        for square in row:
            square.render(window)
    pygame.display.flip()
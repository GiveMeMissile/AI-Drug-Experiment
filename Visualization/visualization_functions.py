import pygame
from Visualization import visual_constants as vc
from Visualization import grid_object


def visual_loop(window):
    pygame.init()
    running = True
    clock = pygame.time.Clock()
    grid = create_grid()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw(window, grid)
        clock.tick(60)


def create_grid():
    x = 0
    y = 100
    grid = []
    for _ in range(10):
        row = []
        x = 0
        print(y)
        for _ in range(20):
            row.append(grid_object.GridObject(vc.GRID_BOX_DIM, x, y, vc.GRID_THICKNESS))
            x += vc.GRID_BOX_DIM - vc.GRID_THICKNESS
            print(x)
        grid.append(row)
        y += vc.GRID_BOX_DIM - vc.GRID_THICKNESS
    return grid


def draw(window, grid):
    window.fill(vc.BLACK)
    for row in grid:
        for square in row:
            square.render(window)
    pygame.display.flip()
import pygame
import numpy as np

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
GREEN = (0, 100, 0)

# Initialize Pygame
pygame.init()

# Set the dimensions of the screen and the grid
screen_width = 500
screen_height = 525

# Set the dimensions and position of the button
button_width = 200
button_height = 25
button_x = 150
button_y = 500

grid_size = 10
num_cols = screen_width // grid_size
num_rows = (screen_height - button_height) // grid_size
screen = pygame.display.set_mode((screen_width, screen_height))

# Set the title of the window
pygame.display.set_caption("Obstacle Map")

# Create an empty occupancy grid map
obstacle_map = np.zeros((num_rows, num_cols), dtype=np.uint8)

# Set the font for the button
font = pygame.font.SysFont(None, 30)



# Set a boolean flag to track if the map is finished and can be saved
map_finished = False

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or map_finished:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            # Check if the mouse is clicked within the grid
            if 0 < mouse_pos[0] < screen_width and \
               0 < mouse_pos[1] < screen_height - button_height:
                col = mouse_pos[0] // grid_size
                row = mouse_pos[1] // grid_size
                # Toggle the occupancy of the grid that was clicked
                if obstacle_map[row, col] == 0:
                    obstacle_map[row, col] = 1
                else:
                    obstacle_map[row, col] = 0
            # Check if the map generation is finished button is clicked
            elif button_x <= mouse_pos[0] <= button_x + button_width and \
                    button_y <= mouse_pos[1] <= button_y + button_height:
                map_finished = True

    # Fill the screen with black
    screen.fill(BLACK)

    # Draw the occupancy grid map
    for row in range(num_rows):
        for col in range(num_cols):
            if obstacle_map[row, col] == 1:
                pygame.draw.rect(screen, GREY, (col*grid_size, row*grid_size, grid_size, grid_size))
            else:
                pygame.draw.rect(screen, WHITE, (col*grid_size, row*grid_size, grid_size, grid_size))

    # Draw the map generation is finished button
    pygame.draw.rect(screen, GREEN, (button_x, button_y, button_width, button_height))
    text = font.render("Map generation finished", True, WHITE)
    screen.blit(text, (button_x + 10, button_y + 10))

    # Check if the map is finished and display a message
    if map_finished:
        message = font.render("Map saved as obstacle_map.npy", True, WHITE)
        screen.blit(message, (screen_width // 2 - 150, screen_height // 2 + 50))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()

import cv2
import matplotlib.pyplot as plt

num_labels, grouped_map = cv2.connectedComponents(obstacle_map, connectivity=4)
np.save("shaped_obstacle_map.npy", grouped_map)

for i in range(num_labels - 1):
    np.sum(grouped_map == i + 1)

print("Number of connected components:", num_labels)
plt.imshow(grouped_map)
plt.show()




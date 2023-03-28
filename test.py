# import pygame
# import test

# pygame.init()
# window = pygame.display.set_mode((400, 400))
# clock = pygame.time.Clock()

# def draw_rect_angle(surface, color, rect, angle, width=0):
#     target_rect = pygame.Rect(rect)
#     shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
#     pygame.draw.rect(shape_surf, color, (0, 0, *target_rect.size), width)
#     rotated_surf = pygame.transform.rotate(shape_surf, angle)
#     surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

# def draw_ellipse_angle(surface, color, rect, angle, width=0):
#     target_rect = pygame.Rect(rect)
#     shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
#     pygame.draw.ellipse(shape_surf, color, (0, 0, *target_rect.size), width)
#     rotated_surf = pygame.transform.rotate(shape_surf, angle)
#     surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

# angle = 00
# run = True
# while run:
#     clock.tick(60)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             run = False

#     window_center = window.get_rect().center

#     window.fill((255, 255, 255))
#     draw_rect_angle(window, (0, 0, 0), (75, 150, 250, 100), angle, 2)
#     draw_ellipse_angle(window, (0, 0, 0), (75, 150, 250, 100), angle, 2)
#     angle += 1
#     pygame.display.flip()

# pygame.quit()
# exit()

import pygame
import numpy as np
from math import degrees

def draw_ellipse_angle(surface, color, rect, angle, width=0):
    target_rect = pygame.Rect(rect)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, color, (0, 0, *target_rect.size), width)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ellipse from 2D covariance matrix")

# Set up the ellipse parameters
mean = np.array([400, 300])  # Center of the ellipse
cov = np.array([[100, 0], [0, 100]])  # 2D covariance matrix

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov)
major_axis = 2 * np.sqrt(5.991 * eigenvalues[0])  # 95% confidence interval for major axis
minor_axis = 2 * np.sqrt(5.991 * eigenvalues[1])  # 95% confidence interval for minor axis
angle = degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))  # Angle between major axis and x-axis

# Draw the ellipse
draw_ellipse_angle(screen, (255, 0, 0), (int(mean[0]-major_axis/2), int(mean[1]-minor_axis/2), int(major_axis), int(minor_axis)), angle, 1)
# pygame.draw.ellipse(screen, (255, 0, 0), (int(mean[0]-major_axis/2), int(mean[1]-minor_axis/2), int(major_axis), int(minor_axis)))

# Display the result
pygame.display.flip()

# Wait for the user to close the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Clean up
pygame.quit()

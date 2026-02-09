"""Rendering utilities for pygame visualisation.

Helper functions for drawing covariance ellipses and star-shaped target
markers on a pygame surface.
"""

from __future__ import annotations

import math

import numpy as np
import pygame


def draw_covariance_ellipse(
    surface: pygame.Surface,
    mean: np.ndarray,
    cov: np.ndarray,
    color: tuple[int, int, int] = (150, 0, 0),
    confidence: float = 5.991,
) -> None:
    """Draw a 2-D covariance ellipse on *surface*.

    Parameters
    ----------
    surface : pygame.Surface
        Target drawing surface.
    mean : np.ndarray
        Centre ``[x, y]`` of the ellipse.
    cov : np.ndarray
        2×2 covariance matrix.
    color : tuple
        RGB colour for the ellipse outline.
    confidence : float
        Chi-squared quantile (default 5.991 ≈ 95 % for 2-DOF).
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    major = 2 * np.sqrt(confidence * eigenvalues[0])
    minor = 2 * np.sqrt(confidence * eigenvalues[1])
    angle = math.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    rect = pygame.Rect(
        int(mean[0] - major / 2 - 2),
        int(mean[1] - minor / 2 - 2),
        int(major + 4),
        int(minor + 4),
    )
    shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, color, (0, 0, *rect.size), 1)
    rotated = pygame.transform.rotate(shape_surf, angle)
    surface.blit(rotated, rotated.get_rect(center=rect.center))


def calculate_star_points(
    center: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    num_points: int = 5,
) -> list[tuple[float, float]]:
    """Compute vertices of a star polygon (e.g. pentagram).

    Parameters
    ----------
    center : tuple
        ``(x, y)`` centre of the star.
    outer_radius : float
        Distance from centre to outer tips.
    inner_radius : float
        Distance from centre to inner vertices.
    num_points : int
        Number of outer tips (default 5).

    Returns
    -------
    list[tuple[float, float]]
        Ordered list of ``(x, y)`` vertices.
    """
    points: list[tuple[float, float]] = []
    for i in range(num_points * 2):
        angle = math.pi / num_points * i
        r = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + r * math.sin(angle)
        y = center[1] - r * math.cos(angle)
        points.append((x, y))
    return points

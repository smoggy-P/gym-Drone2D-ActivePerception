import numpy as np
import math

def get_neighbor(idx, direction, arr):
    directions = {
        'N': (-1, 0),
        'NE': (-1, 1),
        'E': (0, 1),
        'SE': (1, 1),
        'S': (1, 0),
        'SW': (1, -1),
        'W': (0, -1),
        'NW': (-1, -1)
    }

    x, y = idx
    dx, dy = directions[direction]
    x += dx
    y += dy

    if x < 0 or y < 0 or x >= len(arr) or y >= len(arr[0]):
        return None

    return (x, y)

def traversibility(arr, start):
    
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    diagonal_directions = ['NE', 'SE', 'SW', 'NW']
    distances = []

    if arr[start] != 2:
        return 0

    for direction in directions:
        current_pos = start
        distance = 0

        while True:
            next_pos = get_neighbor(current_pos, direction, arr)

            if next_pos is None or arr[next_pos] != 2:
                break

            current_pos = next_pos
            distance += math.sqrt(2) if direction in diagonal_directions else 1

        distances.append(distance)
    # print(distances)

    return np.mean(distances)

arr = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 2, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
])

start = (2, 2)

print(traversibility(arr, start))

import numpy as np
import random
import matplotlib.pyplot as plt


def generate_random_map(map_id, num_clusters):

    np.random.seed(map_id)
    random.seed(map_id)

    # Initialize the grid
    grid_size = 50
    grid = np.zeros((grid_size, grid_size), dtype=int)

    def random_walk(grid, x, y, cluster_index, cluster_size):
        # List of all possible directions to move
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        
        # Mark the current cell as visited by setting it to the cluster index
        grid[x][y] = cluster_index
        cluster_size -= 1

        # While cluster size is not yet met
        while cluster_size > 0:
            # Randomize the order of directions
            np.random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # If the new cell is valid and not visited yet
                if (0 <= nx < grid_size) and (0 <= ny < grid_size) and grid[nx][ny] == 0:
                    x, y = nx, ny
                    grid[x][y] = cluster_index
                    cluster_size -= 1
                    break
            else:
                # No valid cell found, break the loop
                break

    # Generate the clusters
    for cluster_index in range(1, num_clusters + 1):
        while True:
            # Randomly choose a starting cell
            x, y = np.random.randint(5, grid_size - 5, size=(2,))
            
            # If the chosen cell is not yet visited, start a random walk from there
            if grid[x][y] == 0:
                random_walk(grid, x, y, cluster_index, np.random.randint(5, 10))
                break
    
    return grid

# grid = generate_random_map(0, 30, 15)

# plt.imshow(grid, cmap='tab20b')
# plt.show()

# Save the map
# np.save('maps/random_map_{}.npy'.format(map_id), grid)